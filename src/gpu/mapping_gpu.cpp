/**
*
* Obsahuje mapovaci funkce pro buffery pouzite v kodu pro OpenCL.
*
*/

#include "mapping_gpu.h"


/**
* Mapuje neuron vstupni vrstvy v bufferu neural_net_buff
*
* params:
*   i - index neuronu
*/
int kiv_ppr_mapping_gpu::Input_Neuron_I(int i) {
	return i;
}

/**
* Mapuje neuron prvni skryte vrstvy v bufferu neural_net_buff
*
* params:
*   i - index neuronu
*/
int kiv_ppr_mapping_gpu::Hidden1_Neuron_I(int i) {
	return Input_Neuron_I(i) + INPUT_LAYER_NEURONS_COUNT + BIAS;
}

/**
* Mapuje neuron druhe skryte vrstvy v bufferu neural_net_buff
*
* params:
*   i - index neuronu
*/
int kiv_ppr_mapping_gpu::Hidden2_Neuron_I(int i) {
	return Hidden1_Neuron_I(i) + HIDDEN1_LAYER_NEURONS_COUNT + BIAS;
}

/**
* Mapuje neuron vystupni vrstvy v bufferu neural_net_buff
*
* params:
*   i - index neuronu
*/
int kiv_ppr_mapping_gpu::Output_Neuron_I(int i) {
	return Hidden2_Neuron_I(i) + HIDDEN2_LAYER_NEURONS_COUNT + BIAS;
}

/**
* Mapuje vahy neuronu mezi vstupni a prvni skrytou vrstvou.
*
* params:
*   input - index neuronu ve vstupni vrstve
*	hidden1 - index neuronu v prvni skryte vrstve
*/
int kiv_ppr_mapping_gpu::Weight_Input_Hidden1(int input, int hidden1) {
	return 100 + input * HIDDEN1_LAYER_NEURONS_COUNT + hidden1;
}

/**
* Mapuje vahy neuronu mezi prvni skrytou a druhou skrytou vrstvou.
*
* params:
*   hidden1 - index neuronu v prvni skryte vrstve
*	hidden2 - index neuronu ve druhe skryte vrstve
*/
int kiv_ppr_mapping_gpu::Weight_Hidden1_Hidden2(int hidden1, int hidden2) {
	return 270 + hidden1 * HIDDEN2_LAYER_NEURONS_COUNT + hidden2;
}

/**
* Mapuje vahy neuronu mezi druhou skrytou a vystupni vrstvou.
*
* params:
*   hidden2 - index neuronu ve druhe skryte vrstve
*	output - index neuronu ve vystupni vrstve
*/
int kiv_ppr_mapping_gpu::Weight_Hidden2_Output(int hidden2, int output) {
	return 750 + hidden2 * OUTPUT_LAYER_NEURONS_COUNT + output;
}