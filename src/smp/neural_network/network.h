/**
*
* Reprezentuje neuronovou sit (pro verzi SMP).
*
*/

#pragma once

#include	<vector>
#include	"neuron.h"
#include    "tbb/parallel_for.h"
#include	"../../util/utils.h"

namespace kiv_ppr_network {

	/**
	* Struktura reprezentujici neuronovou sit.
	*
	* layers - vrstvy neuronove site - [pocet vrstev][pocet neuronu]
	* relative_errors_vector - vektor relativnich chyb
	*/
	struct TNetwork {
		std::vector<kiv_ppr_neuron::TLayer> layers;
		std::vector<double> relative_errors_vector;
	};

	/**
	* Vytvori novou neuronovou sit.
	*
	* params:
	*   topology - topologie neuronove site
	*
	* return:
	*   nova neuronova sit
	*/
	TNetwork New_Network(const std::vector<unsigned>& topology);

	/**
	* Spousti feed forward.
	*
	* params:
	*   network - neuronova sit
	*   input_values - vektor normalizovanych vstupnich hodnot
	*   training_set_id - identifikator trenovaciho vzorku
	*   input_layer_neurons_count - pocet neuronu vstupni vrstvy
	*/
	void Feed_Forward_Prop(TNetwork& network, const std::vector<double>& input_values, unsigned training_set_id, unsigned input_layer_neurons_count);

	/**
	* Vypocita a ulozi relativni chyby do vektoru relativnich chyb.
	*
	* params:
	*   network - neuronova sit
	*   expected_value - ocekavana hodnota
	*/
	void Save_Relative_Error(TNetwork& network, double expected_value);

	/**
	* Prida vyslednou hodnotu predikce k vektoru vyslednych hodnot.
	*
	* params:
	*   network - neuronova sit
	*	result_values - vektor vyslednych hodnot
	*/
	void Add_Result_Value(TNetwork& network, std::vector<double>& result_values);

	/**
	* Spousti back propagation.
	*
	* params:
	*   network - neuronova sit
	*   target_values - vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty
	*   training_set_id - identifikator trenovaciho vzorku
	*   output_layer_neurons_count - pocet neuronu vystupni vrstvy
	*/
	void Back_Prop(TNetwork& network, const std::vector<double>& target_values, unsigned training_set_id, unsigned output_layer_neurons_count);

	/**
	* Ziska hodnoty neuronu vystupni vrstvy.
	*
	* params:
	*   network - neuronova sit
	*   result_values - vektor hodnot vystupni vrstvy
	*/
	void Get_Results(TNetwork& network, std::vector<double>& result_values);

	/**
	* Vypocita relativni chybu.
	*
	* params:
	*   result_values - vektor hodnot vystupni vrstvy
	*	expected_value - ocekavana hodnota
	*
	* return:
	*	relativn chyba
	*/
	double Calculate_Relative_Error(std::vector<double>& result_values, double expected_value);

	/**
	* Vypocita vyslednou hodnotu.
	*
	* params:
	*   output - vektor hodnot vystupni vrstvy
	*
	* result:
	*	vysledna hodnota
	*/
	double Calculate_Result_Value(std::vector<double>& output);
}
