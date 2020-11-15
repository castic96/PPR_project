#pragma once

#include<vector>
#include "synapse.h"

#define		ETA			0.05	// [0.0 - 1.0] rychlost uèení sítì
#define		ALPHA		0.1		// [0.0 - n] multiplikátor poslední zmìny váhy (momentum)

namespace kiv_ppr_neuron {

	struct TNeuron {
		double output_value;
		std::vector<kiv_ppr_synapse::TSynapse> output_weights;
		unsigned neuron_index;
		double gradient = 0.0;
	};

	struct TLayer {
		std::vector<kiv_ppr_neuron::TNeuron> neurons;
	};

	TNeuron New_Neuron(unsigned number_of_outputs, unsigned neuron_index);
	double Get_Random_Weight();
	void Feed_Forward_Hidden(TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer);
	void Feed_Forward_Output(TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer);
	double Transfer_Function_Hidden(double value);
	double Transfer_Function_Output(double value);
	double Transfer_Function_Hidden_Der(double value);
	double Transfer_Function_Output_Der(double value, double sum);
	void Compute_Output_Gradient(TNeuron& neuron, double target_value, double sum);
	void Compute_Hidden_Gradient(TNeuron& neuron, const TLayer& next_layer);
	void Update_Input_Weight(TNeuron& neuron, TLayer& previous_layer);
	double Sum_Dow(TNeuron& neuron, const TLayer& next_layer);

	TLayer New_Layer();
}