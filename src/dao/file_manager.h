#pragma once

#include	<sstream>
#include	<fstream>
#include	<iostream>
#include	<string>
#include	<vector>
#include	<algorithm>

namespace kiv_ppr_file_manager {
	void Save_Svg_File(std::string file_path, std::string& graph);
	void Save_Ini_File(std::string file_path, std::string& neural_params);
	void Save_Csv_File(std::string file_path, std::string& errors);
	void Save_Results_File(std::string file_path, std::string& results);
	bool Load_Ini_File(std::string file_path, std::vector<std::vector<double>>& loaded_weights, std::vector<unsigned>& neural_network_params);
}

