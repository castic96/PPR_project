#pragma once

#include	<vector>
#include	"neuron.h"
#include    "tbb/parallel_for.h"
#include	"../../util/utils.h"

// Pocet poslednich vysledku, ktere se budou zapocitavat do prumeru
#define		RECENT_AVERAGE_SMOOTHING_FACTOR		100.0

namespace kiv_ppr_network {

	struct TNetwork {
		std::vector<kiv_ppr_neuron::TLayer> layers; // layers[pocet vrstev][pocet neuronu]
		std::vector<double> relative_errors_vector;
	};

	TNetwork New_Network(const std::vector<unsigned>& topology);
	void Feed_Forward_Prop(TNetwork& network, const std::vector<double>& input_values, unsigned training_set_id);
	void Save_Relative_Error(TNetwork& network, double expected_value);
	void Back_Prop(TNetwork& network, const std::vector<double>& target_values, unsigned training_set_id);
	void Get_Results(TNetwork& network, std::vector<double>& result_values);
	double Calculate_Relative_Error(std::vector<double> result_values, double expected_value);

}
