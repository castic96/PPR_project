#include<vector>
#include<cassert>
#include "network.h"


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

	// Spusteni forward propagation
	for (unsigned i = 1; i < network.layers.size(); i++) {

		kiv_ppr_neuron::layer &previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < network.layers[i].neurons.size() - 1; j++) {
			kiv_ppr_neuron::feed_forward(network.layers[i].neurons[j], previous_layer);
		}
	}

}

void kiv_ppr_network::back_prop(kiv_ppr_network::network& network, const std::vector<double>& target_values) {

	// Vypoètení kumulativní chyby (RMS výstupních chyb)
	// TODO: možná zmìnit - máme optimalizovat dle relativní chyby a ne rozdílu !!! viz poznámky!!!
	// abs(vypoèítaná hodnota - namìøená hodnota)/namìøená hodnota - možná zamìnit za výpoèet rozdílu
	
	
	double standard_error = 0.0; 
	double relative_error = 0.0;

	kiv_ppr_neuron::layer& output_layer = network.layers.back();
	//network.standard_error = 0.0;

	for (unsigned i = 0; i < output_layer.neurons.size() - 1; i++) {
		double delta = target_values[i] - output_layer.neurons[i].output_value;

		standard_error += delta * delta;
		//zkontrolovat - èagy to má naopak
		relative_error += abs(output_layer.neurons[i].output_value - target_values[i]) / target_values[i];
	}
	//TODO: tím výpoètem si fakt nejsem jistej..
	standard_error /= output_layer.neurons.size() - 1;
	relative_error /= output_layer.neurons.size() - 1;

	network.error = sqrt(standard_error) + relative_error;

	network.recent_average_error =
		(network.recent_average_error * RECENT_AVERAGE_SMOOTHING_FACTOR + network.error) /
		(RECENT_AVERAGE_SMOOTHING_FACTOR + 1.0);

	// Spocitani gradientu vystupni vrstvy
	for (unsigned i = 0; i < output_layer.neurons.size() - 1; i++) {
		kiv_ppr_neuron::compute_output_gradient(output_layer.neurons[i], target_values[i]);
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