#pragma once

#include<vector>
#include"neuron.h"

// Pocet poslednich vysledku, ktere se budou zapocitavat do prumeru
#define		RECENT_AVERAGE_SMOOTHING_FACTOR		100.0

namespace kiv_ppr_network {

	typedef struct Network {
		std::vector<kiv_ppr_neuron::layer> layers; // layers[pocet vrstev][pocet neuronu]
		double error = 0.0;
		double recent_average_error = 0.0;
	} network;

	network new_network(const std::vector<unsigned>& topology);
	void feed_forward_prop(network& network, const std::vector<double>& input_values);
	void back_prop(network& network, const std::vector<double>& target_values);
	void get_results(network& network, std::vector<double>& result_values);
	double risk_function(const double bg);

}
