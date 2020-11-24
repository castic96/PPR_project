/**
*
* Reprezentuje neuronovou sit (pro verzi SMP).
*
*/

#include<vector>
#include<cassert>
#include "network.h"


/**
* Mapuje index neuronu vstupni vrstvy na vektor vstupnich hodnot.
*
* params:
*	set_num - cislo trenovaciho vzorku
*   n - index neuronu
*	input_layer_neurons_count - pocet neuronu vstupni vrstvy
*/
int Input_Value(int set_num, int n, unsigned input_layer_neurons_count) {
	return set_num * input_layer_neurons_count + n;
}

/**
* Mapuje index neuronu vystupni vrstvy na vektor vystupnich hodnot.
*
* params:
*	set_num - cislo trenovaciho vzorku
*   n - index neuronu
*	input_layer_neurons_count - pocet neuronu vstupni vrstvy
*/
int Target_Value(int set_num, int n, unsigned output_layer_neurons_count) {
	return set_num * output_layer_neurons_count + n;
}

/**
* Vytvori novou neuronovou sit.
*
* params:
*   topology - topologie neuronove site
*
* return:
*   nova neuronova sit
*/
kiv_ppr_network::TNetwork kiv_ppr_network::New_Network(const std::vector<unsigned>& topology) {
	kiv_ppr_network::TNetwork new_network;
	unsigned number_of_layers = topology.size();

	for (unsigned i = 0; i < number_of_layers; i++) {
		new_network.layers.push_back(kiv_ppr_neuron::New_Layer());
		unsigned number_of_outputs;

		if (i == topology.size() - 1) {
			number_of_outputs = 0;
		} 
		else {
			number_of_outputs = topology[i + 1];
		}

		for (unsigned j = 0; j <= topology[i]; j++) {
			new_network.layers.back().neurons.push_back(kiv_ppr_neuron::New_Neuron(number_of_outputs, j));
		}

		new_network.layers.back().neurons.back().output_value = 1.0;

	}

	return new_network;
}

/**
* Spousti feed forward.
*
* params:
*   network - neuronova sit
*   input_values - vektor normalizovanych vstupnich hodnot
*   training_set_id - identifikator trenovaciho vzorku
*   input_layer_neurons_count - pocet neuronu vstupni vrstvy
*/
void kiv_ppr_network::Feed_Forward_Prop(kiv_ppr_network::TNetwork& network, const std::vector<double> &input_values, unsigned training_set_id, unsigned input_layer_neurons_count) {

	// --- Prirazeni vstupnich hodnot do vstupni vrstvy (vstupnich neuronu) ---
	for (unsigned i = 0; i < input_layer_neurons_count; i++) {
		network.layers[0].neurons[i].output_value = input_values[Input_Value(training_set_id, i, input_layer_neurons_count)];
	}

	// --- Spusteni forward propagation pro skryte vrstvy ---
	for (unsigned i = 1; i < network.layers.size() - 1; i++) {

		kiv_ppr_neuron::TLayer& previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < network.layers[i].neurons.size() - 1; j++) {
			kiv_ppr_neuron::Feed_Forward_Hidden(network.layers[i].neurons[j], previous_layer);
		}
	}

	// --- Spusteni forward propagation pro vystupni vrstvu ---
	unsigned last_layer_index = network.layers.size() - 1;
	double sum = 0.0;

	kiv_ppr_neuron::TLayer& previous_layer = network.layers[last_layer_index - 1];

	for (unsigned i = 0; i < network.layers[last_layer_index].neurons.size() - 1; i++) {
		kiv_ppr_neuron::Feed_Forward_Output(network.layers[last_layer_index].neurons[i], previous_layer);
		sum += network.layers[last_layer_index].neurons[i].output_value;
	}

	for (unsigned i = 0; i < network.layers[last_layer_index].neurons.size() - 1; i++) {
		network.layers[last_layer_index].neurons[i].output_value /= sum;
	}

}

/**
* Aktualizuje countery synapsi (pro vykreslovani SVG grafu)
*
* params:
*   network - neuronova sit
*   relative_error - relativni chyba
*/
void Update_Graph_Counters(kiv_ppr_network::TNetwork& network, double relative_error) {

	if (relative_error <= 0.15) {

		tbb::parallel_for(size_t(0), network.layers.size() - 1, [&](size_t i) {
			kiv_ppr_neuron::TLayer& current_layer = network.layers[i];

			for (unsigned j = 0; j < current_layer.neurons.size(); j++) {
				kiv_ppr_neuron::TNeuron& current_neuron = current_layer.neurons[j];

				for (unsigned k = 0; k < current_neuron.output_weights.size(); k++) {
					current_neuron.output_weights[k].counter_blue_graph += current_neuron.output_weights[k].current_counter;
					current_neuron.output_weights[k].counter_green_graph += current_neuron.output_weights[k].current_counter;
					current_neuron.output_weights[k].current_counter = 0.0;
				}

			}
		});

	}
	else {

		tbb::parallel_for(size_t(0), network.layers.size() - 1, [&](size_t i) {
			kiv_ppr_neuron::TLayer& current_layer = network.layers[i];

			for (unsigned j = 0; j < current_layer.neurons.size(); j++) {
				kiv_ppr_neuron::TNeuron& current_neuron = current_layer.neurons[j];

				for (unsigned k = 0; k < current_neuron.output_weights.size(); k++) {
					current_neuron.output_weights[k].counter_green_graph += current_neuron.output_weights[k].current_counter;
					current_neuron.output_weights[k].current_counter = 0.0;
				}

			}
		});

	}
}

/**
* Vypocita a ulozi relativni chyby do vektoru relativnich chyb.
*
* params:
*   network - neuronova sit
*   expected_value - ocekavana hodnota
*/
void kiv_ppr_network::Save_Relative_Error(kiv_ppr_network::TNetwork& network, double expected_value) {
	std::vector<double> result_values;

	kiv_ppr_network::Get_Results(network, result_values);
	double relative_error = kiv_ppr_network::Calculate_Relative_Error(result_values, expected_value);
	network.relative_errors_vector.push_back(relative_error);

	Update_Graph_Counters(network, relative_error);
}

/**
* Prida vyslednou hodnotu predikce k vektoru vyslednych hodnot.
*
* params:
*   network - neuronova sit
*	result_values - vektor vyslednych hodnot
*/
void kiv_ppr_network::Add_Result_Value(kiv_ppr_network::TNetwork& network, std::vector<double>& result_values) {
	std::vector<double> output;

	kiv_ppr_network::Get_Results(network, output);
	double value = kiv_ppr_network::Calculate_Result_Value(output);
	result_values.push_back(value);
}

/**
* Spousti back propagation.
*
* params:
*   network - neuronova sit
*   target_values - vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty
*   training_set_id - identifikator trenovaciho vzorku
*   output_layer_neurons_count - pocet neuronu vystupni vrstvy
*/
void kiv_ppr_network::Back_Prop(kiv_ppr_network::TNetwork& network, const std::vector<double>& target_values, unsigned training_set_id, unsigned output_layer_neurons_count) {
	kiv_ppr_neuron::TLayer& output_layer = network.layers.back();

	// --- Spocitani gradientu vystupni vrstvy ---
	for (unsigned i = 0; i < output_layer.neurons.size() - 1; i++) {
		kiv_ppr_neuron::Compute_Output_Gradient(output_layer.neurons[i], target_values[Target_Value(training_set_id, i, output_layer_neurons_count)]);
	}

	// --- Spocitani gradientu skrytych vrstev ---
	for (unsigned i = network.layers.size() - 2; i > 0; i--) {
		kiv_ppr_neuron::TLayer& hidden_layer = network.layers[i];
		kiv_ppr_neuron::TLayer& next_layer = network.layers[i + 1];

		for (unsigned j = 0; j < hidden_layer.neurons.size(); j++) {
			kiv_ppr_neuron::Compute_Hidden_Gradient(hidden_layer.neurons[j], next_layer);
		}
	}

	// --- Aktualizace vah synapsi - pres vsechny vrstvy od vystupni po vstupni ---
	for (unsigned i = network.layers.size() - 1; i > 0; i--) {
		kiv_ppr_neuron::TLayer& current_layer = network.layers[i];
		kiv_ppr_neuron::TLayer& previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < current_layer.neurons.size() - 1; j++) {
			kiv_ppr_neuron::Update_Input_Weight(current_layer.neurons[j], previous_layer);
		}
	}
}

/**
* Ziska hodnoty neuronu vystupni vrstvy.
*
* params:
*   network - neuronova sit
*   result_values - vektor hodnot vystupni vrstvy
*/
void kiv_ppr_network::Get_Results(kiv_ppr_network::TNetwork& network, std::vector<double>& result_values) {
	result_values.clear();

	for (unsigned i = 0; i < network.layers.back().neurons.size() - 1; i++) {
		result_values.push_back(network.layers.back().neurons[i].output_value);
	}

}

/**
* Vypocita relativni chybu.
*
* params:
*   result_values - vektor hodnot vystupni vrstvy
*	expected_value - ocekavana hodnota
*
* return:
*	relativn chyba
*/
double kiv_ppr_network::Calculate_Relative_Error(std::vector<double>& result_values, double expected_value) {

	unsigned result_index = 0;
	for (unsigned i = 0; i < result_values.size(); i++) {
		if (result_values[i] > result_values[result_index]) {
			result_index = i;
		}
	}

	double result_value = kiv_ppr_utils::Band_Index_To_Level(result_index);

	return (std::abs(result_value - expected_value) / expected_value);
}

/**
* Vypocita vyslednou hodnotu.
*
* params:
*   output - vektor hodnot vystupni vrstvy
*
* result:
*	vysledna hodnota
*/
double kiv_ppr_network::Calculate_Result_Value(std::vector<double>& output) {

	unsigned result_index = 0;
	for (unsigned i = 0; i < output.size(); i++) {
		if (output[i] > output[result_index]) {
			result_index = i;
		}
	}

	return kiv_ppr_utils::Band_Index_To_Level(result_index);
}