#include <random>
#include <cmath>
#include "neuron.h"

kiv_ppr_neuron::TNeuron kiv_ppr_neuron::New_Neuron(unsigned number_of_outputs, unsigned neuron_index) {
	kiv_ppr_neuron::TNeuron new_neuron;

	for (unsigned i = 0; i < number_of_outputs; i++) {
		new_neuron.output_weights.push_back(kiv_ppr_synapse::New_Synapse());
		new_neuron.output_weights.back().weight = Get_Random_Weight();
	}

	new_neuron.neuron_index = neuron_index;

	return new_neuron;
}

kiv_ppr_neuron::TLayer kiv_ppr_neuron::New_Layer() {
	kiv_ppr_neuron::TLayer new_layer;

	return new_layer;
}

double kiv_ppr_neuron::Get_Random_Weight() {
	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_real_distribution<> distribution(0, 1); //uniformni rozdeleni <0,1>

	return distribution(generator);
}

void kiv_ppr_neuron::Feed_Forward_Hidden(kiv_ppr_neuron::TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer) {
	double sum = 0.0;

	for (unsigned i = 0; i < previous_layer.neurons.size(); i++) {
		kiv_ppr_neuron::TNeuron prev_neuron = previous_layer.neurons[i];

		sum += prev_neuron.output_value * prev_neuron.output_weights[neuron.neuron_index].weight;
	}

	neuron.output_value = kiv_ppr_neuron::Transfer_Function_Hidden(sum);
}

void kiv_ppr_neuron::Feed_Forward_Output(kiv_ppr_neuron::TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer) {
	double sum = 0.0;

	for (unsigned i = 0; i < previous_layer.neurons.size(); i++) {
		kiv_ppr_neuron::TNeuron prev_neuron = previous_layer.neurons[i];

		sum += prev_neuron.output_value * prev_neuron.output_weights[neuron.neuron_index].weight;
	}

	neuron.output_value = kiv_ppr_neuron::Transfer_Function_Output(sum);
}

double kiv_ppr_neuron::Transfer_Function_Hidden(double value) {
	return tanh(value);
}

double kiv_ppr_neuron::Transfer_Function_Output(double value) {
	return exp(value);
}

double kiv_ppr_neuron::Transfer_Function_Hidden_Der(double value) {
	double hyperbolic_tan = tanh(value);
	return 1.0 - (hyperbolic_tan * hyperbolic_tan);
}

double kiv_ppr_neuron::Transfer_Function_Output_Der(double value, double sum) {
	// TODO: prozatim pouziju derivaci tangentu..
	/*double f = exp(value) / sum;
	return (f * (1 - f));*/
	return kiv_ppr_neuron::Transfer_Function_Hidden_Der(value);
}

void kiv_ppr_neuron::Compute_Output_Gradient(kiv_ppr_neuron::TNeuron& neuron, double target_value, double sum) {
	double delta = target_value - neuron.output_value;

	neuron.gradient = delta * kiv_ppr_neuron::Transfer_Function_Output_Der(neuron.output_value, sum);
}

void kiv_ppr_neuron::Compute_Hidden_Gradient(kiv_ppr_neuron::TNeuron& neuron, const kiv_ppr_neuron::TLayer& next_layer) {
	double dow = kiv_ppr_neuron::Sum_Dow(neuron, next_layer);

	neuron.gradient = dow * kiv_ppr_neuron::Transfer_Function_Hidden_Der(neuron.output_value);
}

double kiv_ppr_neuron::Sum_Dow(kiv_ppr_neuron::TNeuron& neuron, const kiv_ppr_neuron::TLayer& next_layer) {
	double sum = 0.0;

	for (unsigned i = 0; i < next_layer.neurons.size() - 1; i++) {
		sum += neuron.output_weights[i].weight * next_layer.neurons[i].gradient;
	}

	return sum;
}

void kiv_ppr_neuron::Update_Input_Weight(kiv_ppr_neuron::TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer) {
	for (unsigned i = 0; i < previous_layer.neurons.size(); i++) {
		kiv_ppr_neuron::TNeuron& current_neuron = previous_layer.neurons[i];
		double old_delta_weight = current_neuron.output_weights[neuron.neuron_index].delta_weight;

		double new_delta_weight = ETA * current_neuron.output_value * neuron.gradient + ALPHA * old_delta_weight;

		current_neuron.output_weights[neuron.neuron_index].delta_weight = new_delta_weight;
		current_neuron.output_weights[neuron.neuron_index].weight += new_delta_weight;
	}
}
