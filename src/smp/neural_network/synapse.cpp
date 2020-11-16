#include "synapse.h"

kiv_ppr_synapse::TSynapse kiv_ppr_synapse::New_Synapse() {
	kiv_ppr_synapse::TSynapse new_synapse;
	
	new_synapse.weight = 0.0;
	new_synapse.delta_weight = 0.0;

	return new_synapse;
}