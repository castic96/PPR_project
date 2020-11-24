/**
*
* Reprezentuje synapsi (pro verzi SMP).
*
*/

#pragma once

namespace kiv_ppr_synapse {

	/**
	* Struktura reprezentujici synapsi.
	*
	* weight - vaha synapse
	* delta_weight - vypocitany rozdil vahy
	* current_counter - aktualni counter
	* counter_green_graph - counter pro zeleny graf
	* counter_blue_graph - counter pro modry graf
	*/
	struct TSynapse {
		double weight = 0.0;
		double delta_weight = 0.0;
		double current_counter = 0.0;
		double counter_green_graph = 0.0;
		double counter_blue_graph = 0.0;
	};

	/**
	* Vytvori novou synapsi.
	*
	* return:
	*   nova synapse
	*/
	TSynapse New_Synapse();
}