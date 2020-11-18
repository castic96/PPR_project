#pragma once

#include	<CL/cl2.hpp>

namespace kiv_ppr_network_gpu {

	struct TNetworkGPU {
		cl::Device* default_device;
		cl::CommandQueue* queue;
		cl::Buffer* cl_buff_input_values;
		cl::Buffer* cl_buff_expected_values;
		cl::Buffer* cl_buff_relative_errors;


		std::vector<float> relative_errors_vector;


		std::vector<kiv_ppr_neuron::TLayer> layers; // layers[pocet vrstev][pocet neuronu]
		std::vector<double> relative_errors_vector;
		double error = 0.0;
		double recent_average_error = 0.0;
	};

}