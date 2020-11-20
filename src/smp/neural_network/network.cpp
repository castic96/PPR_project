#include<vector>
#include<cassert>
#include <iostream>
#include "network.h"
#include "../../util/utils.h"


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
			new_network.layers.back().neurons.push_back(kiv_ppr_neuron::New_Neuron(number_of_outputs, j, i));
		}

		new_network.layers.back().neurons.back().output_value = 1.0;

	}

	return new_network;
}

void kiv_ppr_network::Feed_Forward_Prop(kiv_ppr_network::TNetwork& network, const std::vector<double> &input_values) {
	//TODO: možná smazat..
	assert(input_values.size() == network.layers[0].neurons.size() - 1);

	// Prirazeni vstupnich hodnot do vstupni vrstvy (vstupnich neuronu)
	for (unsigned i = 0; i < input_values.size(); i++) {
		network.layers[0].neurons[i].output_value = kiv_ppr_utils::Risk_Function(input_values[i]);
	}

	// Spusteni forward propagation pro skryte vrstvy
	for (unsigned i = 1; i < network.layers.size() - 1; i++) {

		kiv_ppr_neuron::TLayer& previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < network.layers[i].neurons.size() - 1; j++) {
			kiv_ppr_neuron::Feed_Forward_Hidden(network.layers[i].neurons[j], previous_layer);
		}
	}

	// Spusteni forward propagation pro vystupni vrstvu
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

void kiv_ppr_network::Back_Prop(kiv_ppr_network::TNetwork& network, const std::vector<double>& target_values, double expected_value, unsigned counter) {

	// Vypocitani relativni chyby a pridani do vektoru chyb v siti	
	std::vector<double> result_values;
	kiv_ppr_network::Get_Results(network, result_values);
	double relative_error = kiv_ppr_network::Calculate_Relative_Error(result_values, expected_value, counter);
	network.relative_errors_vector.push_back(relative_error);

	// Vypoètení kumulativní chyby (RMS výstupních chyb)
	// TODO: asi k nièemu, pak odstranit
	double standard_error = 0.0;
	kiv_ppr_neuron::TLayer& output_layer = network.layers.back();

	for (unsigned i = 0; i < output_layer.neurons.size() - 1; i++) {
		double delta = target_values[i] - output_layer.neurons[i].output_value;

		standard_error += delta * delta;
	}

	standard_error /= output_layer.neurons.size() - 1;

	network.error = sqrt(standard_error);

	network.recent_average_error =
		(network.recent_average_error * RECENT_AVERAGE_SMOOTHING_FACTOR + network.error) /
		(RECENT_AVERAGE_SMOOTHING_FACTOR + 1.0);

	// Spocitani gradientu vystupni vrstvy
	double sum = 0.0;
	for (unsigned i = 0; i < output_layer.neurons.size() - 1; i++) {
		sum += exp(output_layer.neurons[i].output_value);
	}

	for (unsigned i = 0; i < output_layer.neurons.size() - 1; i++) {
		kiv_ppr_neuron::Compute_Output_Gradient(output_layer.neurons[i], target_values[i], sum);
	}


	//Spocitani gradientu skrytych vrstev
	for (unsigned i = network.layers.size() - 2; i > 0; i--) {
		kiv_ppr_neuron::TLayer& hidden_layer = network.layers[i];
		kiv_ppr_neuron::TLayer& next_layer = network.layers[i + 1];

		for (unsigned j = 0; j < hidden_layer.neurons.size(); j++) {
			kiv_ppr_neuron::Compute_Hidden_Gradient(hidden_layer.neurons[j], next_layer);
		}
	}

	// Aktualizace vah synapsi - pres vsechny vrstvy od vystupni po vstupni
	for (unsigned i = network.layers.size() - 1; i > 0; i--) {
		kiv_ppr_neuron::TLayer& current_layer = network.layers[i];
		kiv_ppr_neuron::TLayer& previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < current_layer.neurons.size() - 1; j++) {
			kiv_ppr_neuron::Update_Input_Weight(current_layer.neurons[j], previous_layer);
		}
	}
}

void kiv_ppr_network::Get_Results(kiv_ppr_network::TNetwork& network, std::vector<double>& result_values) {
	result_values.clear();

	for (unsigned i = 0; i < network.layers.back().neurons.size() - 1; i++) {
		result_values.push_back(network.layers.back().neurons[i].output_value);
	}

}

double kiv_ppr_network::Calculate_Relative_Error(std::vector<double> result_values, double expected_value, unsigned counter) {

	unsigned result_index = 0;
	for (unsigned i = 0; i < result_values.size(); i++) {
		if (result_values[i] > result_values[result_index]) {
			result_index = i;
		}
	}

	std::cout << counter << ": " << result_index << std::endl;

	double result_value = kiv_ppr_utils::Band_Index_To_Level(result_index);

	return (std::abs(result_value - expected_value) / expected_value);
}