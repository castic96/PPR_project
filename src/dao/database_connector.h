#pragma once

#include<vector>
#include <winsqlite/winsqlite3.h>

namespace kiv_ppr_db_connector {

	const int COUNT_OF_INPUT_VALUES = 8;
	const int MEASURE_INTERVAL_MINUTES = 5;

	struct TData_Reader {
		char *db_name;
		sqlite3 *db_handler;
	};

	TData_Reader New_Reader(char* db_name);
	bool Open_Database(TData_Reader& reader);
	bool Close_Database(TData_Reader& reader);
	bool Load_Data(TData_Reader& reader);


	struct TInput {
		std::vector<double> values;
		double expected_value = 0.0;
		int first_id = 0;
		bool valid = false;
	};

	TInput Load_Next(TData_Reader& reader, int last_used_first_id, unsigned prediction_minutes);

}