/**
*
* Hlavni program pro SMP.
*
*/

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

	/**
	* Struktura pro vysledky trenovani neuronove site na SMP.
	*
	* network - neuronova sit
	* weights - vahy synapsi neuronove site
	* neural_ini_str - retezec parametru neuronove site (pro generovani INI souboru)
	* csv_str - retezec relativnich chyb (pro generovani CSV souboru)
	*/
	struct TResults_Training_CPU {
		kiv_ppr_network::TNetwork network;
		std::vector<std::vector<double>> weights;
		std::string neural_ini_str;
		std::string csv_str;
	};
	
	/**
	* Struktura pro vysledky predikce neuronove site na SMP.
	*
	* network - neuronova sit
	* csv_str - retezec relativnich chyb (pro generovani CSV souboru)
	* results - retezec vysledku predikce (pro generovani TXT souboru vysledku)
	*/
	struct TResults_Prediction_CPU {
		kiv_ppr_network::TNetwork network;
		std::string csv_str;
		std::string results;
	};

	/**
	* Spousti trenovani neuronove site na SMP.
	*
	* params:
	*   input_values_risk - vektor normalizovanych vstupnich hodnot
	*   target_values - vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty
	*	expected_values - vektor ocekavanych hodnot
	*
	* return:
	*   vysledky trenovani neuronove site
	*/
	TResults_Training_CPU Run_Training_CPU(std::vector<double>& input_values_risk, 
											std::vector<double>& target_values, 
											std::vector<double>& expected_values);

	/**
	* Spousti predikci na SMP.
	*
	* params:
	*   input_values - vektor vstupnich hodnot
	*   input_values_risk - vektor normalizovanych vstupnich hodnot
	*	expected_values - vektor ocekavanych hodnot
	*	loaded_weights - nactene vahy synapsi pro neuronovou sit
	*	neural_network_params - topologie neuronove site (pocet vrstev a neuronu v kazde vrstve)
	*
	* return:
	*   vysledky predikce 
	*/
	TResults_Prediction_CPU Run_Prediction_CPU(std::vector<double>& input_values,
												std::vector<double>& input_values_risk,
												std::vector<double>& expected_values,
												std::vector<std::vector<double>>& loaded_weights,
												std::vector<unsigned>& neural_network_params);

}