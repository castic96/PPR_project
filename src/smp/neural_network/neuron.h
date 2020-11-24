/**
*
* Reprezentuje jeden neuron site (pro verzi SMP).
*
*/

#pragma once

#include<vector>
#include "synapse.h"

// --- Rychlost uceni site [0.0 - 1.0] ---
#define		ETA			0.05

// --- Multiplikator posledni zmeny vahy (momentum) [0.0 - n] --- 
#define		ALPHA		0.1

namespace kiv_ppr_neuron {

	/**
	* Struktura reprezentujici neuron.
	*
	* output_value - vystupni hodnota neuronu
	* output_weights - synapse vystupujici z neuronu
	* neuron_index - index neuronu
	* gradient - gradient neuronu
	*/
	struct TNeuron {
		double output_value = 0.0;
		std::vector<kiv_ppr_synapse::TSynapse> output_weights;
		unsigned neuron_index = 0;
		double gradient = 0.0;
	};

	/**
	* Struktura reprezentujici vrstvu (sklada se z jednotlivych neuronu).
	*
	* neurons - neurony ve vrstve
	*/
	struct TLayer {
		std::vector<kiv_ppr_neuron::TNeuron> neurons;
	};

	/**
	* Vytvori novy neuron.
	*
	* params:
	*   number_of_outputs - pocet vystupu neuronu
	*   neuron_index - index neuronu
	*
	* return:
	*   novy neuron
	*/
	TNeuron New_Neuron(unsigned number_of_outputs, unsigned neuron_index);

	/**
	* Vytvori novou vrstvu neuronu.
	*
	* return:
	*   nova vrstva
	*/
	TLayer New_Layer();

	/**
	* Feed forward pro skryte vrstvy.
	*
	* params:
	*   neuron - aktualni neuron
	*   previous_layer - predchozi vrstva
	*/
	void Feed_Forward_Hidden(TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer);

	/**
	* Feed forward pro vystupni vrstvu.
	*
	* params:
	*   neuron - aktualni neuron
	*   previous_layer - predchozi vrstva
	*/
	void Feed_Forward_Output(TNeuron& neuron, kiv_ppr_neuron::TLayer& previous_layer);

	/**
	* Aktivacni funkce pro skryte vrstvy.
	*
	* params:
	*   value - vstupni hodnota aktivacni funkce
	*
	* return:
	*	vystupni hodnota aktivacni funkce
	*/
	double Transfer_Function_Hidden(double value);

	/**
	* Aktivacni funkce pro vystupni vrstvu.
	*
	* params:
	*   value - vstupni hodnota aktivacni funkce
	*
	* return:
	*	vystupni hodnota aktivacni funkce
	*/
	double Transfer_Function_Output(double value);

	/**
	* Derivace aktivacni funkce pro skryte vrstvy.
	*
	* params:
	*   value - vstupni hodnota derivace aktivacni funkce
	*
	* return:
	*	vystupni hodnota derivace aktivacni funkce
	*/
	double Transfer_Function_Hidden_Der(double value);

	/**
	* Derivace aktivacni funkce pro vystupni vrstvu.
	*
	* params:
	*   value - vstupni hodnota derivace aktivacni funkce
	*
	* return:
	*	vystupni hodnota derivace aktivacni funkce
	*/
	double Transfer_Function_Output_Der(double value);

	/**
	* Vypocita gradient vystupni vrstvy.
	*
	* params:
	*   neuron - aktualni neuron
	*	target_value - ocekavana hodnota (0 nebo 1)
	*/
	void Compute_Output_Gradient(TNeuron& neuron, double target_value);

	/**
	* Vypocita gradienty pro skryte vrstvy.
	*
	* params:
	*   neuron - aktualni neuron
	*	next_layer - nasledujici vrstva
	*/
	void Compute_Hidden_Gradient(TNeuron& neuron, const TLayer& next_layer);

	/**
	* Vypocita sumu gradientu predchozi vrstvy z vahou synapse.
	*
	* params:
	*   neuron - aktualni neuron
	*	next_layer - nasledujici vrstva
	*/
	double Sum_Dow(TNeuron& neuron, const TLayer& next_layer);

	/**
	* Aktualizuje vahu synapse.
	*
	* params:
	*   neuron - aktualni neuron
	*	previous_layer - predchozi vrstva
	*/
	void Update_Input_Weight(TNeuron& neuron, TLayer& previous_layer);
}