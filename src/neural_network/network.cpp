#include<vector>
#include<cassert>
#include "network.h"


kiv_ppr_network::network kiv_ppr_network::new_network(const std::vector<unsigned>& topology) {
	kiv_ppr_network::network new_network;
	unsigned number_of_layers = topology.size();

	for (unsigned i = 0; i < number_of_layers; i++) {
		new_network.layers.push_back(kiv_ppr_layer::new_layer());
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

	}

	return new_network;
}

void kiv_ppr_network::feed_forward_prop(kiv_ppr_network::network &network, const std::vector<double> &input_values) {
	//TODO: možná smazat..
	assert(input_values.size() == network.layers[0].neurons.size() - 1);

	// Prirazeni vstupnich hodnot do vstupni vrstvy (vstupnich neuronu)
	for (unsigned i = 0; i < input_values.size(); i++) {
		network.layers[0].neurons[i].output_value = input_values[i];
	}

	// Spusteni forward propagation
	for (unsigned i = 1; i < network.layers.size(); i++) {

		kiv_ppr_layer::layer &previous_layer = network.layers[i - 1];

		for (unsigned j = 0; j < network.layers[i].neurons.size() - 1; j++) {
			kiv_ppr_neuron::feed_forward(network.layers[i].neurons[j], previous_layer);
		}
	}


}