#include "file_manager.h"

void kiv_ppr_file_manager::Save_Svg_File(std::string file_path, std::string& graph) {
	std::string suffix = ".svg";

	size_t index = file_path.rfind(suffix);

	if (index != file_path.length() - 4) {
		file_path.append(suffix);
	}

	std::ofstream out(file_path);

	out << graph;

	out.close();
}