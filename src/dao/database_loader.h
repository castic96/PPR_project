/**
*
* Nacte data z databaze a nasledne je parsuje.
*
*/

#pragma once

#include	<vector>
#include    <iostream>
#include	"../constants.h"
#include	"../dao/database_connector.h"
#include	"../util/utils.h"

namespace kiv_ppr_database_loader {

    /**
    * Nacte data z databaze do vektoru elementu.
    *
    * params:
    *   db_name - nazev databaze
    *
    * return:
    *   vektor elementu
    */
	std::vector<kiv_ppr_db_connector::TElement> Load_From_Db(char*& db_name);

    /**
    * Parsuje data z vektoru elementu.
    *
    * params:
    *   input_data - vstupni data, vektor elementu
    *   input_values - vektor vstupnich hodnot
    *   input_values_risk - vektor normalizovanych vstupnich hodnot
    *   target_values - vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty 
    *   expected_values - vektor ocekavanych hodnot
    *   predicted_minutes - pocet predikovanych minut dopredu
    *   input_values_count - pocet trenovacich vzorku
    */
    void Load_Inputs(std::vector<kiv_ppr_db_connector::TElement>& input_data,
                std::vector<double>& input_values, std::vector<double>& input_values_risk, 
                std::vector<double>& target_values, std::vector<double>& expected_values,
                unsigned predicted_minutes, unsigned input_values_count);

}