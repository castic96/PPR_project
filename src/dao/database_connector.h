#pragma once

#include	<vector>
#include	<winsqlite/winsqlite3.h>

namespace kiv_ppr_db_connector {

	struct TData_Reader {
		char *db_name;
		sqlite3 *db_handler;
	};

	TData_Reader New_Reader(char* db_name);
	bool Open_Database(TData_Reader& reader);
	bool Close_Database(TData_Reader& reader);


	struct TElement {
		double ist = 0.0;
		unsigned segment_id = 0; 
	};

	std::vector<TElement> Load_Data(TData_Reader& reader);

}