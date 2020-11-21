#include "svg_generator.h"

kiv_ppr_svg_generator::TSvg_Generator kiv_ppr_svg_generator::New_Generator(kiv_ppr_network::TNetwork& network) {
	kiv_ppr_svg_generator::TSvg_Generator new_generator;

	new_generator.network = network;

	return new_generator;
}

void kiv_ppr_svg_generator::Generate(kiv_ppr_svg_generator::TSvg_Generator& generator, std::string& green_graph, std::string& blue_graph) {

}