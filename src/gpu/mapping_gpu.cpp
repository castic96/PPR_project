#include "mapping_gpu.h"

// ------ Pomocne funkce pro pristup k datum ------------------------------------------------
// ----- BUFFER: neural_net_data -----
// --- Vrstvy neuronu ---
int kiv_ppr_mapping_gpu::input_neuron_i(int i) {
	return i;
}

int kiv_ppr_mapping_gpu::hidden1_neuron_i(int i) {
	return input_neuron_i(i) + INPUT_LAYER_NEURONS_COUNT + BIAS;
}

int kiv_ppr_mapping_gpu::hidden2_neuron_i(int i) {
	return hidden1_neuron_i(i) + HIDDEN1_LAYER_NEURONS_COUNT + BIAS;
}

int kiv_ppr_mapping_gpu::output_neuron_i(int i) {
	return hidden2_neuron_i(i) + HIDDEN2_LAYER_NEURONS_COUNT + BIAS;
}

// --- Vahy mezi neurony (bez biasu) ---
int kiv_ppr_mapping_gpu::weight_input_hidden1(int input, int hidden1) {
	return 100 + input * HIDDEN1_LAYER_NEURONS_COUNT + hidden1;
}

int kiv_ppr_mapping_gpu::weight_hidden1_hidden2(int hidden1, int hidden2) {
	return 270 + hidden1 * HIDDEN2_LAYER_NEURONS_COUNT + hidden2;
}

int kiv_ppr_mapping_gpu::weight_hidden2_output(int hidden2, int output) {
	return 750 + hidden2 * OUTPUT_LAYER_NEURONS_COUNT + output;
}

// ----- BUFFER: helper_data -----
// --- Soucet exponencialnich hodnot vsech neuronu vystupni vrstvy (pro SoftMax) ---
int kiv_ppr_mapping_gpu::exp_sum_output_layer() {
	return 0;
}

// --- Nejvyssi hodnota z neuronu ve vystupni vrstve ---- 
int kiv_ppr_mapping_gpu::max_value_output_layer() {
	return 1;
}

// ----- BUFFER: input_data -----
// --- Pristup k trenovacim datum - vstupy ---
int kiv_ppr_mapping_gpu::input_value(int set_num, int n) {
	return set_num * INPUT_LAYER_NEURONS_COUNT + n;
}


// ----- BUFFER: target_data -----
// --- Pristup k trenovacim datum - ocekavane vstupy ---
int kiv_ppr_mapping_gpu::target_value(int set_num, int n) {
	return set_num * OUTPUT_LAYER_NEURONS_COUNT + n;
}


// ----- BUFFER: result_indexes -----
// --- Pristup k bufferu pro indexy nejvice aktivovanych neuronu vystupni vrstvy ---
int kiv_ppr_mapping_gpu::result_value(int set_num) {
	return set_num;
}


// ----- BUFFER: delta_gradient_data -----
// --- Back propagation ---
int kiv_ppr_mapping_gpu::delta_input_hidden1(int input, int hidden1) {
	return input * HIDDEN1_LAYER_NEURONS_COUNT + hidden1;
}

int kiv_ppr_mapping_gpu::delta_hidden1_hidden2(int hidden1, int hidden2) {
	return 170 + hidden1 * HIDDEN2_LAYER_NEURONS_COUNT + hidden2;
}

int kiv_ppr_mapping_gpu::delta_hidden2_output(int hidden2, int output) {
	return 700 + hidden2 * OUTPUT_LAYER_NEURONS_COUNT + output;
}

int	kiv_ppr_mapping_gpu::error_gradient_hidden1(int i) {
	return 1700 + i;
}

int	kiv_ppr_mapping_gpu::error_gradient_hidden2(int i) {
	return 1750 + i;
}

int kiv_ppr_mapping_gpu::error_gradient_output(int i) {
	return 1800 + i;
}