#pragma once

#include <winsqlite/winsqlite3.h>

namespace kiv_ppr_db_connector {

	typedef struct Data_Reader {
		const char *db_name;
		sqlite3 *db_handler;

	} data_reader;

	data_reader new_reader(const char* db_name);
	bool open_database(data_reader *reader);
	bool close_database(data_reader* reader);
	bool load_data(data_reader* reader);

}