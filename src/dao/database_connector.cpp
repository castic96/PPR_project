#include "database_connector.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <winsqlite/winsqlite3.h>

kiv_ppr_db_connector::data_reader kiv_ppr_db_connector::new_reader(const char* db_name) {

    kiv_ppr_db_connector::data_reader new_reader;

    new_reader.db_name = db_name;

	return new_reader;
}


bool kiv_ppr_db_connector::open_database(data_reader *reader) {

    if (sqlite3_open(reader->db_name, &(reader->db_handler)) != SQLITE_OK) {
        std::cout << "Cannot open database: " << sqlite3_errmsg(reader->db_handler) << std::endl;
        return false;
    }

    else
    {
        std::cout << "Database opened successfully!" << std::endl;
        return true;
    }

}

bool kiv_ppr_db_connector::close_database(data_reader* reader) {

    if (sqlite3_close(reader->db_handler) != SQLITE_OK) {
        std::cout << "Cannot close database: " << sqlite3_errmsg(reader->db_handler) << std::endl;
        return false;
    }

    else
    {
        std::cout << "Database closed successfully!" << std::endl;
        return true;
    }

}

bool kiv_ppr_db_connector::load_data(data_reader* reader) {
    int return_code = 0;
    sqlite3* db_handler = reader->db_handler;
    sqlite3_stmt *statement;
    const char *sql_command = "select measuredat, ist from measuredvalue where segmentid = ? and id >= ? order by id limit ?;";

    if (sqlite3_prepare_v2(db_handler, sql_command, -1, &statement, NULL) != SQLITE_OK) {
        std::cout << "Prepare sql command failure: " << sqlite3_errmsg(db_handler) << std::endl;
        return false;
    }

    if (sqlite3_bind_int(statement, 1, 1) != SQLITE_OK) {
        std::cout << "Bind parameter 'segmentid' failure: " << sqlite3_errmsg(db_handler) << std::endl;
        sqlite3_finalize(statement);
        return false;
    }

    if (sqlite3_bind_int(statement, 2, 1) != SQLITE_OK) {
        std::cout << "Bind parameter 'id' failure: " << sqlite3_errmsg(db_handler) << std::endl;
        sqlite3_finalize(statement);
        return false;
    }

    if (sqlite3_bind_int(statement, 3, 8) != SQLITE_OK) {
        std::cout << "Bind parameter 'limit' failure: " << sqlite3_errmsg(db_handler) << std::endl;
        sqlite3_finalize(statement);
        return false;
    }

    return_code = sqlite3_step(statement);

    while (return_code == SQLITE_ROW) {
        std::cout << "measuredat: " << sqlite3_column_text(statement, 0) << std::endl;
        std::cout << "ist: " << sqlite3_column_double(statement, 1) << std::endl;

        return_code = sqlite3_step(statement);
    }

    if (return_code != SQLITE_DONE) {
        std::cout << "SQL execution error: " << sqlite3_errmsg(db_handler) << std::endl;
        sqlite3_finalize(statement);
        return false;
    }

    sqlite3_finalize(statement);
    return true;

}
