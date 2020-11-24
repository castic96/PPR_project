#include "file_manager.h"

void Save_File(std::string& file_path, std::string& to_write, std::string suffix) {
	size_t index = file_path.rfind(suffix);

	if (index != file_path.length() - 4) {
		file_path.append(suffix);
	}

	std::ofstream out(file_path);

	out << to_write;

	out.close();
}

void kiv_ppr_file_manager::Save_Svg_File(std::string file_path, std::string& graph) {
	std::cout << "> Saving SVG file..." << std::endl;

	std::string suffix = ".svg";
	Save_File(file_path, graph, suffix);

	std::cout << "> Saving SVG file... DONE" << std::endl;
	std::cout << "> SVG file saved to: " << file_path << std::endl;
}

void kiv_ppr_file_manager::Save_Ini_File(std::string file_path, std::string& neural_params) {
	std::cout << "> Saving INI file..." << std::endl;

	std::string suffix = ".ini";
	Save_File(file_path, neural_params, suffix);

	std::cout << "> Saving INI file... DONE" << std::endl;
	std::cout << "> INI file saved to: " << file_path << std::endl;
}

void kiv_ppr_file_manager::Save_Csv_File(std::string file_path, std::string& errors) {
	std::cout << "> Saving CSV file..." << std::endl;

	std::string suffix = ".csv";
	Save_File(file_path, errors, suffix);

	std::cout << "> Saving CSV file... DONE" << std::endl;
	std::cout << "> CSV file saved to: " << file_path << std::endl;
}

void kiv_ppr_file_manager::Save_Results_File(std::string file_path, std::string& results) {
	std::cout << "> Saving TXT results file..." << std::endl;

	std::string suffix = ".txt";
	Save_File(file_path, results, suffix);

	std::cout << "> Saving TXT results file... DONE" << std::endl;
	std::cout << "> TXT results file saved to: " << file_path << std::endl;
}

bool Load_File(std::string file_path, std::ifstream& input_file, std::string suffix) {
	size_t index = file_path.rfind(suffix);

	if (index != file_path.length() - 4) {
		std::cout << "> Specified file '" << file_path << "' does not have required type." << std::endl;
		return false;
	}

	input_file = std::ifstream(file_path);

	if (!input_file.is_open()) {
		std::cout << "> File '" << file_path << "' does not exists." << std::endl;
		return false;
	}

	return true;

}

bool kiv_ppr_file_manager::Load_Ini_File(std::string file_path, std::vector<std::vector<double>>& loaded_weights, std::vector<unsigned>& neural_network_params) {
	std::cout << "> Loading INI file '" << file_path << "' with parameters of neural network..." << std::endl;
	
	std::ifstream input_ini_file;
	std::string suffix = ".ini";

	if (!Load_File(file_path, input_ini_file, suffix)) {
		return false;
	}

	const std::string hidden_layer_lbl("[hidden_layer_");
	const std::string output_layer_lbl("[output_layer]");
	const std::string bias_lbl("bias");
	std::string line;
	unsigned contains_output_layer = 0;
	std::vector<double> current_layer_weights;
	unsigned neuron_index = 0;
	unsigned weight_index = 0;

	// Overeni, zda soubor zacina skrytou vrstvou
	std::getline(input_ini_file, line);

	std::transform(line.begin(), line.end(), line.begin(),
		[](unsigned char c) { return std::tolower(c); });

	if (line.find(hidden_layer_lbl) == std::string::npos) {
		std::cout << "> Specified file '" << file_path << "' does not have required format." << std::endl;
		return false;
	}

	neural_network_params.push_back(1);
	current_layer_weights.clear();

	while (std::getline(input_ini_file, line)) {

		if (line.empty()) {
			continue;
		}

		std::transform(line.begin(), line.end(), line.begin(),
			[](unsigned char c) { return std::tolower(c); });

		if (line.find(hidden_layer_lbl) != std::string::npos) {
			neural_network_params[0]++;
			neural_network_params.push_back(weight_index + 1);
			loaded_weights.push_back(current_layer_weights);
			current_layer_weights.clear();
			neuron_index = 0;
			weight_index = 0;
		}
		else if (line.find(output_layer_lbl) != std::string::npos) {
			neural_network_params[0]++;
			neural_network_params.push_back(weight_index + 1);
			loaded_weights.push_back(current_layer_weights);
			current_layer_weights.clear();
			contains_output_layer++;
			neuron_index = 0;
			weight_index = 0;
		}
		else {
			double weight;
			unsigned current_neuron_index;
			unsigned current_weight_index;

			if (line.find(bias_lbl) != std::string::npos) {

				std::replace_if(line.begin(), line.end(), [](const char& c) { return c != '.' && c != '-' && !std::isdigit(c); }, ' ');

				std::istringstream is_stream(line);

				if (!(is_stream >> current_neuron_index >> weight)) {
					std::cout << "> Error while reading file '" << file_path << "'" << std::endl;
					return false;
				}

			}
			else {

				std::replace_if(line.begin(), line.end(), [](const char& c) { return c != '.' && c != '-' && !std::isdigit(c); }, ' ');

				std::istringstream is_stream(line);

				if (!(is_stream >> current_neuron_index >> current_weight_index >> weight)) {
					std::cout << "> Error while reading file '" << file_path << "'" << std::endl;
					return false;
				}

				if (current_weight_index > weight_index) {
					weight_index = current_weight_index;
				}

			}

			if (current_neuron_index > neuron_index) {
				neuron_index = current_neuron_index;
			}

			current_layer_weights.push_back(weight);

		}

	}

	if (contains_output_layer != 1) {
		return false;
	}

	// Pridam posledni vrstvu do vektoru
	loaded_weights.push_back(current_layer_weights);
	neural_network_params[0]++;
	neural_network_params.push_back(weight_index + 1);
	neural_network_params.push_back(neuron_index + 1);

	std::cout << "> Loading INI file '" << file_path << "' with parameters of neural network... DONE" << std::endl;
	
	return true;
}
