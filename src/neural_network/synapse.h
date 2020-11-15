#pragma once

namespace kiv_ppr_synapse {

	struct TSynapse {
		double weight;
		double delta_weight;
	};

	TSynapse New_Synapse();

}