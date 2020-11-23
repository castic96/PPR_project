#pragma once

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <cassert>
#include	<algorithm>
#include    "../constants.h"
#include    "neural_network/network.h"

namespace kiv_ppr_smp {

	struct TResults_Training_CPU {
		kiv_ppr_network::TNetwork network;
		std::vector<std::vector<double>> weights;
		std::string neural_ini_str;
		std::string csv_str;
	};
	
	struct TResults_Prediction_CPU {
		kiv_ppr_network::TNetwork network;
		std::string csv_str;
		std::string results;
	};

	TResults_Training_CPU Run_Training_CPU(std::vector<double>& input_values_risk, 
											std::vector<double>& target_values, 
											std::vector<double>& expected_values);

	TResults_Prediction_CPU Run_Prediction_CPU(std::vector<double>& input_values,
												std::vector<double>& input_values_risk,
												std::vector<double>& expected_values,
												std::vector<std::vector<double>>& loaded_weights,
												std::vector<unsigned>& neural_network_params);

}