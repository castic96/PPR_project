#pragma once

#include<vector>
#include <winsqlite/winsqlite3.h>

namespace kiv_ppr_db_connector {

	const int COUNT_OF_INPUT_VALUES = 8;
	const int MEASURE_INTERVAL_MINUTES = 5;

	typedef struct Data_Reader {
		const char *db_name;
		sqlite3 *db_handler;
	} data_reader;

	data_reader new_reader(const char* db_name);
	bool open_database(data_reader *reader);
	bool close_database(data_reader* reader);
	bool load_data(data_reader* reader);


	typedef struct Input {
		std::vector<double> values;
		double expected_value;
		int first_id;
		bool valid = false;
	} input;

	input load_next(kiv_ppr_db_connector::data_reader* reader, int last_used_first_id, int prediction_minutes);

}