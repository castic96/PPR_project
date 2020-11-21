#pragma once

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <cassert>
#include    "../constants.h"
#include    "neural_network/network.h"

namespace kiv_ppr_smp {

	struct TResults_CPU {
		kiv_ppr_network::TNetwork network;
		std::vector<double> relative_errors;
		std::vector<double> weights;
	};

	TResults_CPU Run_Training_CPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values);

}