#pragma once

#include	<vector>
#include	"../constants.h"
#include	"../dao/database_connector.h"

namespace kiv_ppr_input_parser {

	struct TInput {
		std::vector<double> values;
		double expected_value = 0.0;
		unsigned first_index = 0;
		bool valid = false;
	};

	size_t Compute_Changed_Index(std::vector<kiv_ppr_db_connector::TElement> input_vector, unsigned first_index, int limit);
	int Compute_Prediction_Places(unsigned prediction_minutes);
	TInput Read_Next(std::vector<kiv_ppr_db_connector::TElement> input_vector, int last_used_first_index, unsigned prediction_minutes);

}