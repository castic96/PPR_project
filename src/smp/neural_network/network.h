#pragma once

#include<vector>
#include"neuron.h"

// Pocet poslednich vysledku, ktere se budou zapocitavat do prumeru
#define		RECENT_AVERAGE_SMOOTHING_FACTOR		100.0

namespace kiv_ppr_network {

	struct TNetwork {
		std::vector<kiv_ppr_neuron::TLayer> layers; // layers[pocet vrstev][pocet neuronu]
		std::vector<double> relative_errors_vector;
		double error = 0.0;
		double recent_average_error = 0.0;
	};

	TNetwork New_Network(const std::vector<unsigned>& topology);
	void Feed_Forward_Prop(TNetwork& network, const std::vector<double>& input_values);
	void Back_Prop(TNetwork& network, const std::vector<double>& target_values, double expected_value, unsigned counter);
	void Get_Results(TNetwork& network, std::vector<double>& result_values);
	double Calculate_Relative_Error(std::vector<double> result_values, double expected_value, unsigned counter);

}
