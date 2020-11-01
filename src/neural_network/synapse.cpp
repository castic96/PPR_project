#include "synapse.h"

kiv_ppr_synapse::synapse kiv_ppr_synapse::new_synapse() {
	kiv_ppr_synapse::synapse new_synapse;
	
	new_synapse.weight = 0.0;
	new_synapse.delta_weight = 0.0;

	return new_synapse;
}