#pragma once

#include<vector>
#include"layer.h"

namespace kiv_ppr_network {

	typedef struct Network {
		std::vector<kiv_ppr_layer::layer> layers; // layers[pocet vrstev][pocet neuronu]
	} network;

	network new_network(const std::vector<unsigned> &topology);
	void feed_forward_prop(network &network, const std::vector<double> &input_values);
	void back_prop(network &network, const std::vector<double> &target_values);
	void feed_forward_prop(network &network, const std::vector<double> &result_values);

}
