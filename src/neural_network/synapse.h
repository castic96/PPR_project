#pragma once

namespace kiv_ppr_synapse {

	typedef struct Synapse {
		double weight;
		double delta_weight;
	} synapse;

	synapse new_synapse();

}