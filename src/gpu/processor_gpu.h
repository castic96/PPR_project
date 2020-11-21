#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <CL/cl2.hpp>
#include	"network_gpu.h"

namespace kiv_ppr_gpu {

	struct TResults_GPU {
		kiv_ppr_network_gpu::TNetworkGPU network;
		std::vector<double> relative_errors;
		std::vector<double> weights;
	};

	TResults_GPU Run_Training_GPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values);
}