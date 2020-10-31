#include <random>
#include <cmath>
#include "neuron.h"

kiv_ppr_neuron::neuron kiv_ppr_neuron::new_neuron(unsigned number_of_outputs, unsigned neuron_index) {
	kiv_ppr_neuron::neuron new_neuron;

	for (unsigned i = 0; i < number_of_outputs; i++) {
		new_neuron.output_weights.push_back(kiv_ppr_synapse::new_synapse());
		new_neuron.output_weights.back().weight = get_random_weight();
	}

	new_neuron.neuron_index = neuron_index;

	return new_neuron;
}

double kiv_ppr_neuron::get_random_weight() {
	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_real_distribution<> distribution(0, 1); //uniformni rozdeleni <0,1>

	return distribution(generator);
}

void kiv_ppr_neuron::feed_forward(kiv_ppr_neuron::neuron& neuron, kiv_ppr_layer::layer& previous_layer) {
	double sum = 0.0;

	for (unsigned i = 0; i < previous_layer.neurons.size(); i++) {
		kiv_ppr_neuron::neuron prev_neuron = previous_layer.neurons[i];

		sum += prev_neuron.output_value * prev_neuron.output_weights[neuron.neuron_index].weight;
	}

	neuron.output_value = kiv_ppr_neuron::transfer_function(sum);
}

double kiv_ppr_neuron::transfer_function(double value) {
	return tanh(value);
}

double kiv_ppr_neuron::transfer_function_derivative(double value) {
	return 1.0 - (tanh(value) * tanh(value));
}