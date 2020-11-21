#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <CL/cl2.hpp>
#include	"network_gpu.h"

namespace kiv_ppr_gpu {
	void Run_Training_GPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values);
}