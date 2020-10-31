#pragma once

#include<vector>
#include "layer.h"
#include "synapse.h"

namespace kiv_ppr_neuron {

	typedef struct Neuron {
		double output_value;
		std::vector<kiv_ppr_synapse::synapse> output_weights;
		unsigned neuron_index;
	} neuron;

	neuron new_neuron(unsigned number_of_outputs, unsigned neuron_index);
	double get_random_weight();
	void feed_forward(neuron &neuron, kiv_ppr_layer::layer &previous_layer);
	double transfer_function(double value);
	double transfer_function_derivative(double value);
}