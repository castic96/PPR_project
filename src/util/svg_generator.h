/**
*
* Generator grafu neuronove site do formatu SVG.
*
*/

#pragma once

#include	<vector>
#include	"../smp/neural_network/network.h"

namespace kiv_ppr_svg_generator {

	// --- Konstanty pro konfiguraci SVG objektu ---
	//
	// --- Sirka platna ---
	const unsigned screen_width = 20000;

	// --- Vyska platna ---
	const unsigned screen_height = 12000;

	// --- Okraj na ose x ---
	const unsigned screen_margin_x = 50;

	// --- Okraj na ose y ---
	const unsigned screen_margin_y = 200;

	// --- Polomer elipsy ---
	const unsigned ellipse_radius = 20;

	// --- Ctvrtina polomeru elipsy ---
	const unsigned ellipse_radius_quarter = ellipse_radius / 4;

	// --- Sirka cary elipsy ---
	const unsigned ellipse_stroke_width = 2;

	// --- Sirka cary synapse ---
	const unsigned line_stroke_width = 1;

	// --- Velikost fontu ---
	const unsigned text_font_size = 10;

	// --- Mezera mezi neurony ---
	const unsigned neurons_space_y = 300;

	// --- Mezera mezi vrstvami ---
	const unsigned layers_space_x = 1500;

	// --- Nejnizsi hodnota barvy ---
	const unsigned graph_min_value = 0;

	// --- Nejvyssi hodnota barvy ---
	const unsigned graph_max_value = 255;
	//
	//

	// --- Zastupci atributu SVG objektu ---
	//
	// --- Zastupce x ---
	const std::string x_proxy = "x_proxy";

	// --- Zastupce x1 ---
	const std::string x1_proxy = "x1_proxy";

	// --- Zastupce x2 ---
	const std::string x2_proxy = "x2_proxy";

	// --- Zastupce y ---
	const std::string y_proxy = "y_proxy";

	// --- Zastupce y1 ---
	const std::string y1_proxy = "y1_proxy";

	// --- Zastupce y2 ---
	const std::string y2_proxy = "y2_proxy";

	// --- Zastupce text ---
	const std::string text_proxy = "text_proxy";

	// --- Zastupce color ---
	const std::string color_proxy = "color_proxy";
	//
	//

	// --- Sablony pro SVG objekty ---
	//
	// --- SVG hlavicka souboru ---
	const std::string svg_header = 
		"<svg width=\"" + std::to_string(screen_width) + "\" height=\"" + std::to_string(screen_height) +
		"\" xmlns=\"http://www.w3.org/2000/svg\">\n\n";

	// --- SVG ukonceni souboru ---
	const std::string svg_footer = "</svg>";

	// --- SVG elipsa ---
	const std::string svg_ellipse =
		"<ellipse ry =\"" + std::to_string(ellipse_radius) +
		"\" rx =\"" + std::to_string(ellipse_radius) +
		"\" cy =\"" + y_proxy +
		"\" cx =\"" + x_proxy +
		"\" stroke-width =\"" + std::to_string(ellipse_stroke_width) + "\" stroke =\"#AAAAAA\" fill =\"#fff\"/>";

	// --- SVG primka ---
	const std::string svg_line =
		"<line stroke-linecap=\"undefined\" stroke-linejoin=\"undefined\" stroke=\"" + color_proxy + "\" y2=\"" + y2_proxy +
		"\" x2=\"" + x2_proxy +
		"\" y1=\"" + y1_proxy +
		"\" x1=\"" + x1_proxy +
		"\" fill-opacity=\"null\" stroke-opacity=\"null\" stroke-width=\"" + std::to_string(line_stroke_width) + "\" fill=\"none\"/>";

	// --- SVG text ---
	const std::string svg_text =
		"<text font-weight=\"bold\" xml:space=\"preserve\" text-anchor=\"start\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"" +
		std::to_string(text_font_size) + "\" y=\"" + y_proxy + "\" x=\"" + x_proxy +
		"\" stroke-width=\"0\" stroke =\"#000\" fill=\"#000000\">" + text_proxy + "</text>";

	// --- SVG mezera ---
	const std::string line_space = "\n\n";
	//
	//

	/**
	* Struktura SVG generatoru.
	*
	* network - neuronova sit
	* min_counter_green_graph - minimalni cislo counteru pro zeleny graf
	* max_counter_green_graph - maximalni cislo counteru pro zeleny graf
	* min_counter_blue_graph - minimalni cislo counteru pro modry graf
	* max_counter_blue_graph - maximalni cislo counteru pro modry graf
	* max_neurons - nejvyssi pocet neuronu ve vrstve
	* frame - kostra SVG souboru
	* svg_synapses - vektor synapsi
	* svg_texts - vektor textu
	*/
	struct TSvg_Generator {
		kiv_ppr_network::TNetwork network;
		double min_counter_green_graph;
		double max_counter_green_graph;
		double min_counter_blue_graph;
		double max_counter_blue_graph;
		unsigned max_neurons;
		std::string frame;
		std::vector<std::vector<std::string>> svg_synapses;
		std::vector<std::vector<std::string>> svg_texts;
	};

	/**
	* Vytvori novy SVG generator.
	*
	* params:
	*   network - neuronova sit
	*
	* return:
	*   novy SVG generator
	*/
	TSvg_Generator New_Generator(kiv_ppr_network::TNetwork& network);

	/**
	* Vygeneruje SVG grafy neuronove site.
	*
	* params:
	*   generator - SVG generator
	*   green_graph - zeleny graf ve formatu SVG
	*   blue_graph - modry graf ve formatu SVG
	*/
	void Generate(kiv_ppr_svg_generator::TSvg_Generator& generator, std::string& green_graph, std::string& blue_graph);
}