/**
*
* Reprezentuje synapsi (pro verzi SMP).
*
*/

#include "synapse.h"

/**
* Vytvori novou synapsi.
*
* return:
*   nova synapse
*/
kiv_ppr_synapse::TSynapse kiv_ppr_synapse::New_Synapse() {
	kiv_ppr_synapse::TSynapse new_synapse;
	
	new_synapse.weight = 0.0;
	new_synapse.delta_weight = 0.0;
	new_synapse.current_counter = 0.0;
	new_synapse.counter_green_graph = 0.0;
	new_synapse.counter_blue_graph = 0.0;

	return new_synapse;
}