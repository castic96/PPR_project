#pragma once

#include	"../smp/neural_network/network.h"

namespace kiv_ppr_svg_generator {

	struct TSvg_Generator {
		kiv_ppr_network::TNetwork network;
	};

	TSvg_Generator New_Generator(kiv_ppr_network::TNetwork& network);
	void Generate(kiv_ppr_svg_generator::TSvg_Generator& generator, std::string& green_graph, std::string& blue_graph);

}