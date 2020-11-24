/**
*
* Hlavni program pro GPU.
*
*/

#pragma once

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include	"network_gpu.h"
#include    <CL/cl2.hpp>

namespace kiv_ppr_gpu {

	/**
	* Struktura pro vysledky trenovani neuronove site na GPU.
	*
	* network - neuronova sit
	* relative_errors - vektor relativnich chyb
	* neural_ini_str - retezec parametru neuronove site (pro generovani INI souboru)
	* csv_str - retezec relativnich chyb (pro generovani CSV souboru)
	*/
	struct TResults_Training_GPU {
		kiv_ppr_network_gpu::TNetworkGPU network;
		std::vector<double> relative_errors;
		std::string neural_ini_str;
		std::string csv_str;
	};

	/**
	* Spousti trenovani neuronove site na GPU.
	*
	* params:
	*   input_values - vektor normalizovanych vstupnich hodnot
	*   target_values - vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty
	*	expected_values - vektor ocekavanych hodnot
	*
	* return:
	*   vysledky trenovani neuronove site
	*/
	TResults_Training_GPU Run_Training_GPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values);
}