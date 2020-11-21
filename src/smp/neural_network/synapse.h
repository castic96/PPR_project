#pragma once

namespace kiv_ppr_synapse {

	struct TSynapse {
		double weight;
		double delta_weight;
		double current_counter;
		double counter_green_graph;
		double counter_blue_graph;
	};

	TSynapse New_Synapse();

}