/**
*
* Obsahuje mapovaci funkce pro buffery pouzite v kodu pro OpenCL.
*
*/

#pragma once

// --- Verze OpenCL ---
#define		CL_HPP_TARGET_OPENCL_VERSION		200

// --- Velikost bufferu pro ulozeni neuronove site ---
#define		CL_BUFF_NEURAL_NET_DATA_SIZE		1700

// --- Velikost bufferu pro ulozeni delt a gradientu neuronove site ---
#define		CL_BUFF_DELTA_GRADIENT_DATA_SIZE	2000

// --- Neuron neprezentujici bias ---
#define     BIAS								1

// --- Velikost pomocneho bufferu ---
#define     HELPER_DATA_BUFF_SIZE				10

#include "../constants.h"

namespace kiv_ppr_mapping_gpu {

	/**
	* Mapuje neuron vstupni vrstvy v bufferu neural_net_buff
	*
	* params:
	*   i - index neuronu
	*/
	int Input_Neuron_I(int i);

	/**
	* Mapuje neuron prvni skryte vrstvy v bufferu neural_net_buff
	*
	* params:
	*   i - index neuronu
	*/
	int Hidden1_Neuron_I(int i);

	/**
	* Mapuje neuron druhe skryte vrstvy v bufferu neural_net_buff
	*
	* params:
	*   i - index neuronu
	*/
	int Hidden2_Neuron_I(int i);

	/**
	* Mapuje neuron vystupni vrstvy v bufferu neural_net_buff
	*
	* params:
	*   i - index neuronu
	*/
	int Output_Neuron_I(int i);

	/**
	* Mapuje vahy neuronu mezi vstupni a prvni skrytou vrstvou.
	*
	* params:
	*   input - index neuronu ve vstupni vrstve
	*	hidden1 - index neuronu v prvni skryte vrstve
	*/
	int Weight_Input_Hidden1(int input, int hidden1);

	/**
	* Mapuje vahy neuronu mezi prvni skrytou a druhou skrytou vrstvou.
	*
	* params:
	*   hidden1 - index neuronu v prvni skryte vrstve
	*	hidden2 - index neuronu ve druhe skryte vrstve
	*/
	int Weight_Hidden1_Hidden2(int hidden1, int hidden2);

	/**
	* Mapuje vahy neuronu mezi druhou skrytou a vystupni vrstvou.
	*
	* params:
	*   hidden2 - index neuronu ve druhe skryte vrstve
	*	output - index neuronu ve vystupni vrstve
	*/
	int Weight_Hidden2_Output(int hidden2, int output);

}