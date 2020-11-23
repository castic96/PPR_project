#include "file_manager.h"

void Save_File(std::string file_path, std::string& to_write, std::string suffix) {
	size_t index = file_path.rfind(suffix);

	if (index != file_path.length() - 4) {
		file_path.append(suffix);
	}

	std::ofstream out(file_path);

	out << to_write;

	out.close();
}

void kiv_ppr_file_manager::Save_Svg_File(std::string file_path, std::string& graph) {
	std::string suffix = ".svg";

	Save_File(file_path, graph, suffix);
}

void kiv_ppr_file_manager::Save_Ini_File(std::string file_path, std::string& neural_params) {
	std::string suffix = ".ini";

	Save_File(file_path, neural_params, suffix);
}

void kiv_ppr_file_manager::Save_Csv_File(std::string file_path, std::string& errors) {
	std::string suffix = ".csv";

	Save_File(file_path, errors, suffix);
}