/**
*
* Reprezentuje neuronovou sit (pro verzi GPU).
*
*/

#pragma once

#include	<vector>
#include    <iostream>
#include	<fstream>
#include	<sstream>
#include    "../util/utils.h"
#include	"mapping_gpu.h"
#include	<CL/cl2.hpp>

namespace kiv_ppr_network_gpu {

	/**
	* Struktura reprezentujici neuronovou sit.
	*/
	struct TNetworkGPU {

		// --- Zarizeni pro vypocet ---
		cl::Device* default_device;

		// --- Fronta pozadavku ---
		cl::CommandQueue* queue;

		// --- Vektor vyslednych hodnot ---
		std::vector<cl_float> result_values;

		// --- Sit nactena validne ---
		bool is_valid = true;

		// --- Pocet trenovacich vzorku --- 
		unsigned num_of_training_sets = 0;

		// --- CL buffer pro ulozeni hodnot neuronu a vah ---
		cl::Buffer* cl_buff_neural_net_data;

		// --- CL buffer pro ulozeni delt a gradientu neuronove site ---
		cl::Buffer* cl_buff_delta_gradient_data;

		// --- CL buffer pro vstupni hodnoty trenovaci mnoziny ---
		cl::Buffer* cl_buff_input_data;

		// --- CL buffer pro ocekavane hodnoty trenovaci mnoziny ---
		cl::Buffer* cl_buff_target_data;

		// --- CL buffer pro pomocne vypocty na zarizeni ---
		cl::Buffer* cl_buff_helper_data;

		// --- CL buffer pro ulozeni indexu neuronu vystupni vrstvy s nejvyssi hodnotou ---
		cl::Buffer* cl_buff_result_indexes;

		// --- CL buffer pro ulozeni indexu aktualni trenovaci mnoziny ---
		cl::Buffer* cl_buff_training_set_id;

		// --- Pole neuronu a vah --- 
		cl_float* neural_net_buff;

		// --- Pole delt a gradientu neuronove site ---
		cl_float* delta_gradient_buff;

		// --- Pole vstupnich hodnot trenovaci mnoziny ---
		cl_float* input_values_buff;

		// --- Pole ocekavanych hodnot trenovaci mnoziny (hodnoty 0 a 1, 
		// kde 1 je na miste ocekavane hodnoty) ---
		cl_float* target_values_buff;

		// --- Pole pro pomocne vypocty na zarizeni ---
		cl_float* helper_buff;

		// --- Pole pro ulozeni indexu neuronu vystupni vrstvy s nejvyssi hodnotou ---
		cl_int* result_indexes_buff;

		// --- Pole pro ulozeni indexu aktualni trenovaci mnoziny ---
		cl_int* training_set_id_buff;

		// --- Kernel pro zvyseni pocitadla trenovaci mnoziny ---
		cl::Kernel* inc_train_set_id;
		
		// --- Kernel pro prirazeni pocatecnich hodnot do vsech vrstev ---
		cl::Kernel* set_data_to_layers;

		// --- Kernel pro feed forward mezi vstupni vrstvou a prvni skrytou vrstvou ---
		cl::Kernel* feed_forward_input_hidden1;

		// --- Kernel pro feed forward mezi prvni a druhou skrytou vrstvou ---
		cl::Kernel* feed_forward_hidden1_hidden2;

		// --- Kernel pro feed forward mezi druhou skrytou vrstvou a vystupni vrstvou ---
		cl::Kernel* feed_forward_hidden2_output;

		// --- Kernel pro zjisteni indexu neuronu vystupni vrstvy s nejvyssi hodnotou a ulozeni do bufferu ---
		cl::Kernel* set_index_of_result;

		// --- Kernel pro back propagation - pro vystupni vrstvu ---
		cl::Kernel* back_prop_output;

		// --- Kernel pro back propagation - pro druhou skrytou vrstvu ---
		cl::Kernel* back_prop_hidden2;

		// --- Kernel pro back propagation - pro prvni skrytou vrstvu ---
		cl::Kernel* back_prop_hidden1;

		// --- Kernel pro back propagation - pro vstupni vrstvu ---
		cl::Kernel* back_prop_input;

		// --- Kernel pro update vah synapsi ---
		cl::Kernel* update_weights;
	};

	/**
	* Vytvori novou neuronovou sit.
	*
	* params:
	*   input_values - vektor normalizovanych vstupnich hodnot
	*   target_values - vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty
	*   num_of_training_sets - pocet trenovacich vzorku
	*
	* return:
	*   nova neuronova sit
	*/
	kiv_ppr_network_gpu::TNetworkGPU New_Network(std::vector<double>& input_values, std::vector<double>& target_values, unsigned num_of_training_sets);

	/**
	* Inicializuje pocatecni data pro neuronovou sit.
	*
	* params:
	*   network - neuronova sit
	*   input_values_size - velikost pole vstupnich hodnot
	*   target_values_size - velikost pole ocekavanych hodnot
	*/
	void Init_Data(TNetworkGPU& network, unsigned input_values_size, unsigned target_values_size);

	/**
	* Spousti trenovani neuronove site.
	*
	* params:
	*   network - neuronova sit
	*/
	void Train(TNetworkGPU& network);

	/**
	* Ziska vektor relativnich chyb.
	*
	* params:
	*   network - neuronova sit
	*   expected_values - vektor ocekavanych hodnot
	*   relative_errors_vector - vektor relativnich chyb
	*/
	void Get_Relative_Errors_Vector(TNetworkGPU& network, std::vector<double>& expected_values, std::vector<double>& relative_errors_vector);

	/**
	* Uvolni pamet po bufferech a polich.
	*
	* params:
	*   network - neuronova sit
	*/
	void Clean(TNetworkGPU& network);

}