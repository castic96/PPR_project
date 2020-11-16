#pragma once

#include	<vector>
#include	"../../dao/database_connector.h"

namespace kiv_ppr_input_parser {

	const unsigned COUNT_OF_INPUT_VALUES = 8;
	const unsigned MEASURE_INTERVAL_MINUTES = 5;


	struct TInput {
		std::vector<double> values;
		double expected_value = 0.0;
		unsigned first_index = 0;
		bool valid = false;
	};

	TInput Read_Next(std::vector<kiv_ppr_db_connector::TElement> input_vector, int last_used_first_index, unsigned prediction_minutes);

}