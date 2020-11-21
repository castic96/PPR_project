#pragma once

#define		CL_HPP_TARGET_OPENCL_VERSION		200
#define		CL_BUFF_NEURAL_NET_DATA_SIZE		1700
#define		CL_BUFF_DELTA_GRADIENT_DATA_SIZE	2000
#define     BIAS								1
#define     HELPER_DATA_BUFF_SIZE				10

#include "../constants.h"

namespace kiv_ppr_mapping_gpu {

// ------ Pomocne funkce pro pristup k datum ------------------------------------------------
// ----- BUFFER: neural_net_data -----
// --- Vrstvy neuronu ---
	int input_neuron_i(int i);
	int hidden1_neuron_i(int i);
	int hidden2_neuron_i(int i);
	int output_neuron_i(int i);

	// --- Vahy mezi neurony (bez biasu) ---
	int weight_input_hidden1(int input, int hidden1);
	int weight_hidden1_hidden2(int hidden1, int hidden2);
	int weight_hidden2_output(int hidden2, int output);

	// ----- BUFFER: helper_data -----
	// --- Soucet exponencialnich hodnot vsech neuronu vystupni vrstvy (pro SoftMax) ---
	int exp_sum_output_layer();

	// --- Nejvyssi hodnota z neuronu ve vystupni vrstve ---- 
	int max_value_output_layer();

	// ----- BUFFER: input_data -----
	// --- Pristup k trenovacim datum - vstupy ---
	int input_value(int set_num, int n);


	// ----- BUFFER: target_data -----
	// --- Pristup k trenovacim datum - ocekavane vstupy ---
	int target_value(int set_num, int n);


	// ----- BUFFER: result_indexes -----
	// --- Pristup k bufferu pro indexy nejvice aktivovanych neuronu vystupni vrstvy ---
	int result_value(int set_num);


	// ----- BUFFER: delta_gradient_data -----
	// --- Back propagation ---
	int delta_input_hidden1(int input, int hidden1);
	int delta_hidden1_hidden2(int hidden1, int hidden2);
	int delta_hidden2_output(int hidden2, int output);
	int	error_gradient_hidden1(int i);
	int	error_gradient_hidden2(int i);
	int error_gradient_output(int i);

}