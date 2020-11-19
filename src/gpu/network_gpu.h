#pragma once

#include	<vector>
#include    <iostream>
#include	<fstream>
#include	<sstream>
#include    "../util/utils.h"
#include	"mapping_gpu.h"
#include	<CL/cl2.hpp>

namespace kiv_ppr_network_gpu {

	struct TNetworkGPU {

		// Zarizeni pro vypocet
		cl::Device* default_device;

		// Fronta pozadavku
		cl::CommandQueue* queue;

		// Vektor vyslednych hodnot
		std::vector<cl_float> result_values;

		// Sit nactena validne
		bool is_valid = true;

		unsigned num_of_training_sets = 0;

		// ----- BUFFERY --------------------------------------------------------------------
		// Buffer pro ulozeni hodnot neuronu a vah
		cl::Buffer* cl_buff_neural_net_data;

		// Buffer pro ulozeni delt a gradientu neuronove site
		cl::Buffer* cl_buff_delta_gradient_data;

		// Vstupni hodnoty trenovaci mnoziny
		cl::Buffer* cl_buff_input_data;

		// Ocekavane hodnoty trenovaci mnoziny
		cl::Buffer* cl_buff_target_data;

		// Pomocny buffer pro vypocty na zarizeni
		cl::Buffer* cl_buff_helper_data;

		// Buffer pro ulozeni indexu neuronu vystupni vrstvy s nejvyssi hodnotou
		cl::Buffer* cl_buff_result_indexes;

		// Buffer pro ulozeni indexu aktualni trenovaci mnoziny
		cl::Buffer* cl_buff_training_set_id;
		// ----------------------------------------------------------------------------------

		// ----- DATA------------------------------------------------------------------------
		cl_float* neural_net_buff;
		cl_float* delta_gradient_buff;
		cl_float* input_values_buff;
		cl_float* target_values_buff;
		cl_float* helper_buff;
		cl_int* result_indexes_buff;
		cl_int* training_set_id_buff;
		// ----------------------------------------------------------------------------------

		// ----- KERNELY --------------------------------------------------------------------
		// Prirazeni pocatecnich hodnot do vsech vrstev 
		cl::Kernel* set_data_to_layers;

		// Feed Forward mezi vstupni vrstvou a prvni skrytou vrstvou
		cl::Kernel* feed_forward_input_hidden1;

		// Feed Forward mezi prvni a druhou skrytou vrstvou
		cl::Kernel* feed_forward_hidden1_hidden2;

		// Feed Forward mezi druhou skrytou vrstvou a vystupni vrstvou
		cl::Kernel* feed_forward_hidden2_output;

		// Zjisteni indexu neuronu vystupni vrstvy s nejvyssi hodnotou a ulozeni do bufferu
		cl::Kernel* set_index_of_result;

		// Back propagation - pro vystupni vrstvu
		cl::Kernel* back_prop_output;

		// Back propagation - pro druhou skrytou vrstvu
		cl::Kernel* back_prop_hidden2;

		// Back propagation - pro prvni skrytou vrstvu
		cl::Kernel* back_prop_hidden1;

		// Back propagation - pro vstupni vrstvu
		cl::Kernel* back_prop_input;

		// Update vah synapsi
		cl::Kernel* update_weights;
		// ----------------------------------------------------------------------------------
	};

	TNetworkGPU New_Network(std::vector<cl_float> input_values, std::vector<cl_float> expected_values, unsigned num_of_training_sets);
	void Train(TNetworkGPU network);

}