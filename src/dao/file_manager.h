/**
*
* Nacita data ze souboru a uklada data do souboru.
*
*/

#pragma once

#include	<sstream>
#include	<fstream>
#include	<iostream>
#include	<string>
#include	<vector>
#include	<algorithm>

namespace kiv_ppr_file_manager {

	/**
	* Uklada SVG soubor s grafem.
	*
	* params:
	*   file_path - cesta k vystupnimu souboru
	*   graph - data k ulozeni
	*/
	void Save_Svg_File(std::string file_path, std::string& graph);

	/**
	* Uklada INI soubor s parametry neuronove site.
	*
	* params:
	*   file_path - cesta k vystupnimu souboru
	*   neural_params - data k ulozeni
	*/
	void Save_Ini_File(std::string file_path, std::string& neural_params);

	/**
	* Uklada CSV soubor s relativnimi chybami.
	*
	* params:
	*   file_path - cesta k vystupnimu souboru
	*   errors - data k ulozeni
	*/
	void Save_Csv_File(std::string file_path, std::string& errors);

	/**
	* Uklada TXT soubor s vysledky predikce.
	*
	* params:
	*   file_path - cesta k vystupnimu souboru
	*   results - data k ulozeni
	*/
	void Save_Results_File(std::string file_path, std::string& results);

	/**
	* Nacita INI soubor s vahami neuronove site.
	*
	* params:
	*   file_path - cesta k vystupnimu souboru
	*   loaded_weights - cesta ke vstupnimu souboru
	*	neural_network_params - vahy neuronove site
	*/
	bool Load_Ini_File(std::string file_path, std::vector<std::vector<double>>& loaded_weights, std::vector<unsigned>& neural_network_params);
}

