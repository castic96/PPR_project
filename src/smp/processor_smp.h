#pragma once

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <cassert>
#include    "tbb/parallel_for.h"
#include    "../constants.h"
#include    "neural_network/network.h"

namespace kiv_ppr_smp {

	void Run_Training_CPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values);

}