#pragma once

#include	<vector>
#include	"../smp/neural_network/network.h"

namespace kiv_ppr_svg_generator {

	// --- Konstanty pro konfiguraci SVG objektu ---
	const unsigned screen_width = 20000;
	const unsigned screen_height = 12000;
	const unsigned screen_margin_x = 50;
	const unsigned screen_margin_y = 200;
	const unsigned ellipse_radius = 20;
	const unsigned ellipse_radius_quarter = ellipse_radius / 4;
	const unsigned ellipse_stroke_width = 2;
	const unsigned line_stroke_width = 1;
	const unsigned text_font_size = 11;
	const unsigned neurons_space_y = 300;
	const unsigned layers_space_x = 1500;
	const unsigned color_max_value = 255;

	// --- Zastupci atributu SVG objektu ---
	const std::string x_proxy = "x_proxy";
	const std::string x1_proxy = "x1_proxy";
	const std::string x2_proxy = "x2_proxy";
	const std::string y_proxy = "y_proxy";
	const std::string y1_proxy = "y1_proxy";
	const std::string y2_proxy = "y2_proxy";
	const std::string text_proxy = "text_proxy";
	const std::string color_proxy = "color_proxy";

	// Sablony pro SVG objekty
	const std::string svg_header = 
		"<svg width=\"" + std::to_string(screen_width) + "\" height=\"" + std::to_string(screen_height) +
		"\" xmlns=\"http://www.w3.org/2000/svg\">\n\n";

	const std::string svg_footer = "</svg>";

	const std::string svg_ellipse =
		"<ellipse ry =\"" + std::to_string(ellipse_radius) +
		"\" rx =\"" + std::to_string(ellipse_radius) +
		"\" cy =\"" + y_proxy +
		"\" cx =\"" + x_proxy +
		"\" stroke-width =\"" + std::to_string(ellipse_stroke_width) + "\" stroke =\"#AAAAAA\" fill =\"#fff\"/>";

	const std::string svg_line =
		"<line stroke-linecap=\"undefined\" stroke-linejoin=\"undefined\" stroke=\"" + color_proxy + "\" y2=\"" + y2_proxy +
		"\" x2=\"" + x2_proxy +
		"\" y1=\"" + y1_proxy +
		"\" x1=\"" + x1_proxy +
		"\" fill-opacity=\"null\" stroke-opacity=\"null\" stroke-width=\"" + std::to_string(line_stroke_width) + "\" fill=\"none\"/>";

	const std::string svg_text =
		"<text xml:space=\"preserve\" text-anchor=\"start\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"" + 
		std::to_string(text_font_size) + "\" y=\"" + y_proxy + "\" x=\"" + x_proxy +
		"\" stroke-width=\"0\" stroke =\"#000\" fill=\"#000000\">" + text_proxy + "</text>";

	const std::string line_space = "\n\n";

	struct TSvg_Generator {
		kiv_ppr_network::TNetwork network;
		double max_counter_green_graph;
		double max_counter_blue_graph;
		unsigned max_neurons;
		std::string frame;
		std::vector<std::vector<std::string>> svg_synapses;
		std::vector<std::vector<std::string>> svg_texts;
	};

	TSvg_Generator New_Generator(kiv_ppr_network::TNetwork& network);
	void Generate(kiv_ppr_svg_generator::TSvg_Generator& generator, std::string& green_graph, std::string& blue_graph);

}