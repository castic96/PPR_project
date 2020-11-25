/**
*
* Generator grafu neuronove site do formatu SVG.
*
*/

#include "svg_generator.h"


/**
* Vytvori novy SVG generator.
*
* params:
*   network - neuronova sit
*
* return:
*   novy SVG generator
*/
kiv_ppr_svg_generator::TSvg_Generator kiv_ppr_svg_generator::New_Generator(kiv_ppr_network::TNetwork& network) {
	kiv_ppr_svg_generator::TSvg_Generator new_generator;

	new_generator.network = network;
	new_generator.min_counter_green_graph = 0.0;
	new_generator.max_counter_green_graph = 0.0;
	new_generator.min_counter_blue_graph = 0.0;
	new_generator.max_counter_blue_graph = 0.0;
	new_generator.max_neurons = 0;

	return new_generator;
}

/**
* Nastavi hodnotu do SVH objektu misto zastupce.
*
* params:
*   svg_object - SVG objekt
*   old_value - zastupce
*   new_value - hodnota
*
* return:
*	true, pokud se hodnota nastavila spravne
*/
bool Set_Svg_Value(std::string& svg_object, const std::string& old_value, const std::string& new_value) {
	size_t start = svg_object.find(old_value);

	if (start == std::string::npos) {
		return false;
	}
		
	svg_object.replace(start, old_value.length(), new_value);

	return true;
}

/**
* Zjisti maximalni hodnoty counteru a nejvyssi pocet neuronu ve vrstve.
*
* params:
*   generator - SVG generator
*/
void Initialize_Max_Values(kiv_ppr_svg_generator::TSvg_Generator& generator) {
	std::vector<kiv_ppr_neuron::TLayer> layers = generator.network.layers;

	double min_counter_green_graph = 0.0;
	double max_counter_green_graph = 0.0;
	double min_counter_blue_graph = 0.0;
	double max_counter_blue_graph = 0.0;
	unsigned max_neurons = 0;
	unsigned neurons_count;

	for (unsigned i = 0; i < layers.size() - 1; i++) {
		kiv_ppr_neuron::TLayer& current_layer = layers[i];
		neurons_count = current_layer.neurons.size() - 1;

		if (neurons_count > max_neurons) {
			max_neurons = neurons_count;
		}

		for (unsigned j = 0; j < neurons_count; j++) {
			kiv_ppr_neuron::TNeuron& current_neuron = current_layer.neurons[j];

			for (unsigned k = 0; k < current_neuron.output_weights.size(); k++) {
				double current_counter_green_graph = current_neuron.output_weights[k].counter_green_graph;
				double current_counter_blue_graph = current_neuron.output_weights[k].counter_blue_graph;

				if (current_counter_green_graph < min_counter_green_graph) {
					min_counter_green_graph = current_counter_green_graph;
				}

				if (current_counter_green_graph > max_counter_green_graph) {
					max_counter_green_graph = current_counter_green_graph;
				}

				if (current_counter_blue_graph < min_counter_blue_graph) {
					min_counter_blue_graph = current_counter_blue_graph;
				}

				if (current_counter_blue_graph > max_counter_blue_graph) {
					max_counter_blue_graph = current_counter_blue_graph;
				}

			}

		}

	}

	neurons_count = layers[layers.size() - 1].neurons.size() - 1;

	if (neurons_count > max_neurons) {
		max_neurons = neurons_count;
	}

	generator.min_counter_blue_graph = min_counter_blue_graph;
	generator.max_counter_blue_graph = max_counter_blue_graph;
	generator.min_counter_green_graph = min_counter_green_graph;
	generator.max_counter_green_graph = max_counter_green_graph;
	generator.max_neurons = max_neurons;

}

/**
* Naskaluje souradnici y pro vrstvu.
*
* params:
*   neurons_count - pocet neuronu ve vrstve
*   max_neurons - nejvyssi pocet neuronu ve vrstve
* return:
*	naskalovana hodnota souradnice y pro vrstvu
*/
unsigned Scale_Layers_Y(unsigned neurons_count, unsigned max_neurons) {
	double norm = neurons_count / (double)max_neurons;

	unsigned half_neurons = max_neurons / 2;
	return static_cast<unsigned int>(half_neurons - (half_neurons * norm)) * kiv_ppr_svg_generator::neurons_space_y;
}

/**
* Naskaluje souradnici x pro text.
*
* params:
*   neurons_count - pocet neuronu ve vrstve
*   max_neurons - nejvyssi pocet neuronu ve vrstve
* return:
*	naskalovana hodnota souradnice x pro text
*/
unsigned Scale_Texts_Y(unsigned neurons_count, unsigned max_neurons) {
	double norm = neurons_count / (double)max_neurons;

	unsigned half_neurons = max_neurons / 2;
	return static_cast<unsigned int>(((half_neurons * norm)) * kiv_ppr_svg_generator::text_font_size);
}

/**
* Vygeneruje SVG synapsi dle zadanych souradnic.
*
* params:
*   x1 - souradnice x bodu 1
*   y1 - souradnice y bodu 1
*   x2 - souradnice x bodu 2
*   y2 - souradnice y bodu 2
* return:
*	SVG synapse v podobe retezce
*/
std::string Generate_Svg_Synapse(unsigned x1, unsigned y1, unsigned x2, unsigned y2) {
	std::string svg_synapse = kiv_ppr_svg_generator::svg_line;

	Set_Svg_Value(svg_synapse, kiv_ppr_svg_generator::x1_proxy, std::to_string(x1));
	Set_Svg_Value(svg_synapse, kiv_ppr_svg_generator::y1_proxy, std::to_string(y1));
	Set_Svg_Value(svg_synapse, kiv_ppr_svg_generator::x2_proxy, std::to_string(x2));
	Set_Svg_Value(svg_synapse, kiv_ppr_svg_generator::y2_proxy, std::to_string(y2));

	return svg_synapse;
}

/**
* Vygeneruje SVG text dle zadanych souradnic.
*
* params:
*   x - souradnice x
*   y - souradnice y
* return:
*	SVG text v podobe retezce
*/
std::string  Generate_Svg_Text(unsigned x, unsigned y) {
	std::string svg_text = kiv_ppr_svg_generator::svg_text;

	Set_Svg_Value(svg_text, kiv_ppr_svg_generator::x_proxy, std::to_string(x));
	Set_Svg_Value(svg_text, kiv_ppr_svg_generator::y_proxy, std::to_string(y));

	return svg_text;
}

/**
* Vygeneruje SVG neuron dle zadanych souradnic.
*
* params:
*   x - souradnice x
*   y - souradnice y
* return:
*	SVG neuron v podobe retezce
*/
std::string Generate_Svg_Neuron(unsigned x, unsigned y) {
	std::string svg_neuron = kiv_ppr_svg_generator::svg_ellipse;

	Set_Svg_Value(svg_neuron, kiv_ppr_svg_generator::x_proxy, std::to_string(x));
	Set_Svg_Value(svg_neuron, kiv_ppr_svg_generator::y_proxy, std::to_string(y));

	return svg_neuron;
}

/**
* Naskaluje hodnotu do rozmezi 'graph_min_value' - 'graph_max_value'.
*
* params:
*   count - hodnota counteru
*   min_value - minimalni hodnota counteru
*   max_value - maximalni hodnota counteru
* return:
*	naskalovana hodnota v danem rozmezi
*/
unsigned Scale_Value(double count, double min_value, double max_value) {
	return static_cast<unsigned int>((count - min_value) *
		(kiv_ppr_svg_generator::graph_max_value - kiv_ppr_svg_generator::graph_min_value) /
		(max_value - min_value) + kiv_ppr_svg_generator::graph_min_value);
}

/**
* Vygeneruje SVG kostru pro grafy neuronove site.
*
* params:
*   generator - SVG generator
*/
void Generate_Frame(kiv_ppr_svg_generator::TSvg_Generator& generator) {
	std::vector<kiv_ppr_neuron::TLayer>& layers = generator.network.layers;
	std::vector<std::vector<std::string>>& svg_synapses = generator.svg_synapses;
	std::vector<std::vector<std::string>>& svg_texts = generator.svg_texts;
	std::string& frame = generator.frame;
	unsigned max_neurons = generator.max_neurons;
	std::vector<std::string> current_synapses;
	std::vector<std::string> current_texts;

	unsigned x_init = kiv_ppr_svg_generator::screen_margin_x;
	unsigned y_init;
	unsigned y_init_prev;

	unsigned x = x_init;
	unsigned y;

	for (unsigned i = 0; i < layers.size(); i++) {
		kiv_ppr_neuron::TLayer& current_layer = layers[i];
		unsigned current_neurons_count = current_layer.neurons.size() - 1;

		y_init = kiv_ppr_svg_generator::screen_margin_y + Scale_Layers_Y(current_neurons_count, max_neurons);
		y = y_init;

		current_synapses.clear();
		current_texts.clear();

		for (unsigned j = 0; j < current_neurons_count; j++) {
			std::string svg_neuron = Generate_Svg_Neuron(x, y);

			frame.append(svg_neuron);
			frame.append(kiv_ppr_svg_generator::line_space);

			if (i > 0) {

				kiv_ppr_neuron::TLayer& previous_layer = layers[i - 1];
				unsigned previous_neurons_count = previous_layer.neurons.size() - 1;

				unsigned y_text = y - Scale_Texts_Y(previous_neurons_count, max_neurons) + (kiv_ppr_svg_generator::text_font_size);

				unsigned x1 = x - kiv_ppr_svg_generator::ellipse_radius;
				unsigned y1 = y;
				unsigned x2 = x1 - kiv_ppr_svg_generator::layers_space_x + 2 * kiv_ppr_svg_generator::ellipse_radius;
				unsigned y2 = y_init_prev;

				for (unsigned k = 0; k < previous_neurons_count; k++) {

					// --- Synapse ---
					std::string svg_synapse = Generate_Svg_Synapse(x1, y1, x2, y2);
					current_synapses.push_back(svg_synapse);

					// --- Text ---
					std::string svg_text = Generate_Svg_Text(x1 + kiv_ppr_svg_generator::ellipse_radius_quarter, 
															 y_text + (k * kiv_ppr_svg_generator::text_font_size));
					current_texts.push_back(svg_text);

					y2 += kiv_ppr_svg_generator::neurons_space_y;
				}

			}

			y += kiv_ppr_svg_generator::neurons_space_y;

		}

		if (i > 0) {
			svg_synapses.push_back(current_synapses);
			svg_texts.push_back(current_texts);
		}

		x += kiv_ppr_svg_generator::layers_space_x;
		y_init_prev = y_init;

	}

}

/**
* Prida do kostry grafu synapse a texty a vygeneruje grafy neuronove site.
*
* params:
*   generator - SVG generator
*   green_graph - zeleny graf ve formatu SVG
*   blue_graph - modry graf ve formatu SVG
*/
void Generate_Graphs(kiv_ppr_svg_generator::TSvg_Generator& generator, std::string& green_graph, std::string& blue_graph) {
	std::cout << "> Generating SVG file with graphs..." << std::endl;

	std::vector<kiv_ppr_neuron::TLayer>& layers = generator.network.layers;
	std::vector<std::vector<std::string>>& svg_synapses = generator.svg_synapses;
	std::vector<std::vector<std::string>>& svg_texts = generator.svg_texts;

	green_graph.append(generator.frame);
	blue_graph.append(generator.frame);

	for (unsigned i = 1; i < layers.size(); i++) {
		kiv_ppr_neuron::TLayer& current_layer = layers[i];
		unsigned current_neurons_count = current_layer.neurons.size() - 1;

		kiv_ppr_neuron::TLayer& previous_layer = layers[i - 1];
		unsigned previous_neurons_count = previous_layer.neurons.size() - 1;

		for (unsigned j = 0; j < current_neurons_count; j++) {

			for (unsigned k = 0; k < previous_neurons_count; k++) {

				double counter_green_graph = previous_layer.neurons[k].output_weights[j].counter_green_graph;
				double counter_blue_graph = previous_layer.neurons[k].output_weights[j].counter_blue_graph;
				unsigned green_value = Scale_Value(counter_green_graph, generator.min_counter_green_graph, generator.max_counter_green_graph);
				unsigned blue_value = Scale_Value(counter_blue_graph, generator.min_counter_blue_graph, generator.max_counter_blue_graph);

				// --- Barva synapse ---
				std::string green_color = "rgb(0, " + std::to_string(green_value) + ", 0)";
				std::string blue_color = "rgb(0, 0," + std::to_string(blue_value) + ")";

				std::string green_line = generator.svg_synapses[i - 1][k + j * previous_neurons_count];
				std::string blue_line = generator.svg_synapses[i - 1][k + j * previous_neurons_count];

				Set_Svg_Value(green_line, kiv_ppr_svg_generator::color_proxy, green_color);
				Set_Svg_Value(blue_line, kiv_ppr_svg_generator::color_proxy, blue_color);

				green_graph.append(green_line);
				blue_graph.append(blue_line);

				green_graph.append(kiv_ppr_svg_generator::line_space);
				blue_graph.append(kiv_ppr_svg_generator::line_space);

				// --- Text synapse ---
				std::string green_text = generator.svg_texts[i - 1][k + j * previous_neurons_count];
				std::string blue_text = generator.svg_texts[i - 1][k + j * previous_neurons_count];

				Set_Svg_Value(green_text, kiv_ppr_svg_generator::text_proxy, std::to_string(green_value));
				Set_Svg_Value(blue_text, kiv_ppr_svg_generator::text_proxy, std::to_string(blue_value));

				green_graph.append(green_text);
				blue_graph.append(blue_text);

				green_graph.append(kiv_ppr_svg_generator::line_space);
				blue_graph.append(kiv_ppr_svg_generator::line_space);
			}

		}

	}

	std::cout << "> Generating SVG file with graphs... DONE" << std::endl;

}

/**
* Vygeneruje SVG grafy neuronove site.
*
* params:
*   generator - SVG generator
*   green_graph - zeleny graf ve formatu SVG
*   blue_graph - modry graf ve formatu SVG
*/
void kiv_ppr_svg_generator::Generate(kiv_ppr_svg_generator::TSvg_Generator& generator, std::string& green_graph, std::string& blue_graph) {
	generator.frame.append(kiv_ppr_svg_generator::svg_header);
	
	Initialize_Max_Values(generator);
	Generate_Frame(generator);
	Generate_Graphs(generator, green_graph, blue_graph);

	green_graph.append(kiv_ppr_svg_generator::svg_footer);
	blue_graph.append(kiv_ppr_svg_generator::svg_footer);
}