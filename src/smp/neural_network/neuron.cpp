/**
*
* Reprezentuje jeden neuron site (pro verzi SMP).
*
*/

#include <cmath>
#include "../../util/utils.h"
#include "neuron.h"


/**
* Vytvori novy neuron.
*
* params:
*   number_of_outputs - pocet vystupu neuronu
*   neuron_index - index neuronu
*
* return:
*   novy neuron
*/
kiv_ppr_neuron::TNeuron kiv_ppr_neuron::New_Neuron(unsigned number_of_outputs, unsigned neuron_index) {
	kiv_ppr_neuron::TNeuron new_neuron;

	for (unsigned i = 0; i < number_of_outputs; i++) {
		new_neuron.output_weights.push_back(kiv_ppr_synapse::New_Synapse());
		new_neuron.output_weights.back().weight = kiv_ppr_utils::Get_Random_Weight();
	}

	new_neuron.neuron_index = neuron_index;

	return new_neuron;
}

/**
* Vytvori novou vrstvu neuronu.
*
* return:
*   nova vrstva
*/
kiv_ppr_neuron::TLayer kiv_ppr_neuron::New_Layer() {
	kiv_ppr_neuron::TLayer new_layer;

	return new_layer;
}

/**
* Feed forward pro skryte vrstvy.
*
* params:
*   neuron - aktualni neuron
*   previous_layer - predchozi vrstva
*/
void kiv_ppr_neuron::Feed_Forward_Hidden(kiv_ppr_neuron::TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer) {
	double sum = 0.0;

	for (unsigned i = 0; i < previous_layer.neurons.size(); i++) {
		kiv_ppr_neuron::TNeuron& prev_neuron = previous_layer.neurons[i];

		sum += prev_neuron.output_value * prev_neuron.output_weights[neuron.neuron_index].weight;
		prev_neuron.output_weights[neuron.neuron_index].current_counter = prev_neuron.output_value * prev_neuron.output_weights[neuron.neuron_index].weight;
	
	}

	neuron.output_value = kiv_ppr_neuron::Transfer_Function_Hidden(sum);
}

/**
* Feed forward pro vystupni vrstvu.
*
* params:
*   neuron - aktualni neuron
*   previous_layer - predchozi vrstva
*/
void kiv_ppr_neuron::Feed_Forward_Output(kiv_ppr_neuron::TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer) {
	double sum = 0.0;

	for (unsigned i = 0; i < previous_layer.neurons.size(); i++) {
		kiv_ppr_neuron::TNeuron& prev_neuron = previous_layer.neurons[i];

		sum += prev_neuron.output_value * prev_neuron.output_weights[neuron.neuron_index].weight;
		prev_neuron.output_weights[neuron.neuron_index].current_counter = prev_neuron.output_value * prev_neuron.output_weights[neuron.neuron_index].weight;

	}

	neuron.output_value = kiv_ppr_neuron::Transfer_Function_Output(sum);
}

/**
* Aktivacni funkce pro skryte vrstvy.
*
* params:
*   value - vstupni hodnota aktivacni funkce
*
* return:
*	vystupni hodnota aktivacni funkce
*/
double kiv_ppr_neuron::Transfer_Function_Hidden(double value) {
	return tanh(value);
}

/**
* Aktivacni funkce pro vystupni vrstvu.
*
* params:
*   value - vstupni hodnota aktivacni funkce
*
* return:
*	vystupni hodnota aktivacni funkce
*/
double kiv_ppr_neuron::Transfer_Function_Output(double value) {
	return exp(value);
}

/**
* Derivace aktivacni funkce pro skryte vrstvy.
*
* params:
*   value - vstupni hodnota derivace aktivacni funkce
*
* return:
*	vystupni hodnota derivace aktivacni funkce
*/
double kiv_ppr_neuron::Transfer_Function_Hidden_Der(double value) {
	double hyperbolic_tan = tanh(value);
	return 1.0 - (hyperbolic_tan * hyperbolic_tan);
}

/**
* Derivace aktivacni funkce pro vystupni vrstvu.
*
* params:
*   value - vstupni hodnota derivace aktivacni funkce
*
* return:
*	vystupni hodnota derivace aktivacni funkce
*/
double kiv_ppr_neuron::Transfer_Function_Output_Der(double value) {
	return kiv_ppr_neuron::Transfer_Function_Hidden_Der(value);
}

/**
* Vypocita gradient vystupni vrstvy.
*
* params:
*   neuron - aktualni neuron
*	target_value - ocekavana hodnota (0 nebo 1)
*/
void kiv_ppr_neuron::Compute_Output_Gradient(kiv_ppr_neuron::TNeuron& neuron, double target_value) {
	double delta = target_value - neuron.output_value;

	neuron.gradient = delta * kiv_ppr_neuron::Transfer_Function_Output_Der(neuron.output_value);
}

/**
* Vypocita gradienty pro skryte vrstvy.
*
* params:
*   neuron - aktualni neuron
*	next_layer - nasledujici vrstva
*/
void kiv_ppr_neuron::Compute_Hidden_Gradient(kiv_ppr_neuron::TNeuron& neuron, const kiv_ppr_neuron::TLayer& next_layer) {
	double dow = kiv_ppr_neuron::Sum_Dow(neuron, next_layer);

	neuron.gradient = dow * kiv_ppr_neuron::Transfer_Function_Hidden_Der(neuron.output_value);
}

/**
* Vypocita sumu gradientu predchozi vrstvy z vahou synapse.
*
* params:
*   neuron - aktualni neuron
*	next_layer - nasledujici vrstva
*/
double kiv_ppr_neuron::Sum_Dow(kiv_ppr_neuron::TNeuron& neuron, const kiv_ppr_neuron::TLayer& next_layer) {
	double sum = 0.0;

	for (unsigned i = 0; i < next_layer.neurons.size() - 1; i++) {
		sum += neuron.output_weights[i].weight * next_layer.neurons[i].gradient;
	}

	return sum;
}

/**
* Aktualizuje vahu synapse.
*
* params:
*   neuron - aktualni neuron
*	previous_layer - predchozi vrstva
*/
void kiv_ppr_neuron::Update_Input_Weight(kiv_ppr_neuron::TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer) {
	for (unsigned i = 0; i < previous_layer.neurons.size(); i++) {
		kiv_ppr_neuron::TNeuron& current_neuron = previous_layer.neurons[i];
		double old_delta_weight = current_neuron.output_weights[neuron.neuron_index].delta_weight;

		double new_delta_weight = ETA * current_neuron.output_value * neuron.gradient + ALPHA * old_delta_weight;

		current_neuron.output_weights[neuron.neuron_index].delta_weight = new_delta_weight;
		current_neuron.output_weights[neuron.neuron_index].weight += new_delta_weight;
	}
}