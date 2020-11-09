#include<vector>
#include<cassert>
#include "network.h"
#include "../util/mapping.h"


kiv_ppr_network::network kiv_ppr_network::new_network(const std::vector<unsigned>& topology) {
	kiv_ppr_network::network new_network;
	unsigned number_of_layers = topology.size();

	for (unsigned i = 0; i < number_of_layers; i++) {
		new_network.layers.push_back(kiv_ppr_neuron::new_layer());
		unsigned number_of_outputs;

		if (i == topology.size() - 1) {
			number_of_outputs = 0;
		} 
		else {
			number_of_outputs = topology[i + 1];
		}

		for (unsigned j = 0; j <= topology[i]; j++) {
			new_network.layers.back().neurons.push_back(kiv_ppr_neuron::new_neuron(number_of_outputs, j));
		}

		new_network.layers.back().neurons.back().output_value = 1.0;

	}

	return new_network;
}

void kiv_ppr_network::feed_forward_prop(kiv_ppr_network::network &network, const std::vector<double> &input_values) {
	//TODO: možná smazat..
	assert(input_values.size() == network.layers[0].neurons.size() - 1);

	// Prirazeni vstupnich hodnot do vstupni vrstvy (vstupnich neuronu)
	for (unsigned i = 0; i < input_values.size(); i++) {
		network.layers[0].neurons[i].output_value = risk_function(input_values[i]);
	}

	// Spusteni forward propagation pro skryte vrstvy
	for (unsigned i = 1; i < network.layers.size() - 1; i++) {

		kiv_ppr_neuron::layer &previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < network.layers[i].neurons.size() - 1; j++) {
			kiv_ppr_neuron::feed_forward_hidden(network.layers[i].neurons[j], previous_layer);
		}
	}

	// Spusteni forward propagation pro vystupni vrstvu
	unsigned last_layer_index = network.layers.size() - 1;
	double sum = 0.0;

	kiv_ppr_neuron::layer& previous_layer = network.layers[last_layer_index - 1];

	for (unsigned i = 0; i < network.layers[last_layer_index].neurons.size() - 1; i++) {
		kiv_ppr_neuron::feed_forward_output(network.layers[last_layer_index].neurons[i], previous_layer);
		sum += network.layers[last_layer_index].neurons[i].output_value;
	}

	for (unsigned i = 0; i < network.layers[last_layer_index].neurons.size() - 1; i++) {
		network.layers[last_layer_index].neurons[i].output_value /= sum;
	}

}

void kiv_ppr_network::back_prop(kiv_ppr_network::network& network, const std::vector<double>& target_values, double expected_value) {

	// Vypocitani relativni chyby a pridani do vektoru chyb v siti	
	std::vector<double> result_values;
	kiv_ppr_network::get_results(network, result_values);
	double relative_error = kiv_ppr_network::calculate_relative_error(result_values, expected_value);
	network.relative_errors_vector.push_back(relative_error);

	// Vypoètení kumulativní chyby (RMS výstupních chyb)
	// TODO: asi k nièemu, pak odstranit
	double standard_error = 0.0;
	kiv_ppr_neuron::layer& output_layer = network.layers.back();

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
		kiv_ppr_neuron::compute_output_gradient(output_layer.neurons[i], target_values[i], sum);
	}


	//Spocitani gradientu skrytych vrstev
	for (unsigned i = network.layers.size() - 2; i > 0; i--) {
		kiv_ppr_neuron::layer& hidden_layer = network.layers[i];
		kiv_ppr_neuron::layer& next_layer = network.layers[i + 1];

		for (unsigned j = 0; j < hidden_layer.neurons.size(); j++) {
			kiv_ppr_neuron::compute_hidden_gradient(hidden_layer.neurons[j], next_layer);
		}
	}

	// Aktualizace vah synapsi - pres vsechny vrstvy od vystupni po vstupni
	for (unsigned i = network.layers.size() - 1; i > 0; i--) {
		kiv_ppr_neuron::layer& current_layer = network.layers[i];
		kiv_ppr_neuron::layer& previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < current_layer.neurons.size() - 1; j++) {
			kiv_ppr_neuron::update_input_weight(current_layer.neurons[j], previous_layer);
		}
	}
}

double kiv_ppr_network::risk_function(const double bg) {
	// DOI:  10.1080/10273660008833060
	const double original_risk = 1.794 * (pow(log(bg), 1.026) - 1.861);    //mmol/L

	return original_risk / 3.5;
}

void kiv_ppr_network::get_results(kiv_ppr_network::network& network, std::vector<double>& result_values) {
	result_values.clear();

	for (unsigned i = 0; i < network.layers.back().neurons.size() - 1; i++) {
		result_values.push_back(network.layers.back().neurons[i].output_value);
	}

}

double kiv_ppr_network::calculate_relative_error(std::vector<double> result_values, double expected_value) {

	unsigned result_index = 0;
	for (unsigned i = 0; i < result_values.size(); i++) {
		if (result_values[i] > result_values[result_index]) {
			result_index = i;
		}
	}

	double result_value = kiv_ppr_mapping::band_index_to_level(result_index);

	return (std::abs(result_value - expected_value) / expected_value);
}