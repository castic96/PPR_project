#pragma once

#include<vector>
#include "synapse.h"

#define		ETA			0.15	// [0.0 - 1.0] rychlost uèení sítì
#define		ALPHA		0.5		// [0.0 - n] multiplikátor poslední zmìny váhy (momentum)

namespace kiv_ppr_neuron {

	typedef struct Neuron {
		double output_value;
		std::vector<kiv_ppr_synapse::synapse> output_weights;
		unsigned neuron_index;
		double gradient;
	} neuron;

	typedef struct Layer {
		std::vector<kiv_ppr_neuron::neuron> neurons;
	} layer;

	neuron new_neuron(unsigned number_of_outputs, unsigned neuron_index);
	double get_random_weight();
	void feed_forward(neuron &neuron, kiv_ppr_neuron::layer &previous_layer);
	double transfer_function_hidden(double value);
	double transfer_function_hidden_der(double value);
	void compute_output_gradient(neuron& neuron, double target_value);
	void compute_hidden_gradient(neuron& neuron, const layer& next_layer);
	void update_input_weight(neuron& neuron, layer& previous_layer);
	double sum_dow(neuron& neuron, const layer& next_layer);

	layer new_layer();
}