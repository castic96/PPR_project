#pragma once

#include	<vector>
#include	"../constants.h"
#include	"../dao/database_connector.h"
#include	"../util/utils.h"

namespace kiv_ppr_database_loader {

	std::vector<kiv_ppr_db_connector::TElement> Load_From_Db(char*& db_name);
    void Load_Inputs(std::vector<kiv_ppr_db_connector::TElement>& input_data,
                std::vector<double>& input_values, std::vector<double>& input_values_risk, 
                std::vector<double>& target_values, std::vector<double>& expected_values,
                unsigned predicted_minutes, unsigned input_values_count);

}