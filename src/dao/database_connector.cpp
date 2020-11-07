#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <winsqlite/winsqlite3.h>

#include "database_connector.h"

kiv_ppr_db_connector::data_reader kiv_ppr_db_connector::new_reader(char* db_name) {

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

kiv_ppr_db_connector::input create_new_input(std::vector<int> ids, std::vector<double> values, bool is_valid) {
    kiv_ppr_db_connector::input new_input;

    if (ids.empty() || !is_valid) {
        new_input.valid = false;
    }
    else {
        new_input.valid = true;

        for (int i = 0; i < kiv_ppr_db_connector::COUNT_OF_INPUT_VALUES; i++) {
            new_input.values.push_back(values[i]);
        }

        new_input.expected_value = values[values.size() - 1];
        new_input.first_id = ids[0];
    }

    return new_input;
}

size_t compute_changed_index(std::vector<int> segments_id) {

    for (size_t i = 0; i < segments_id.size() - 1; i++) {
        if (segments_id[i] != segments_id[i + 1]) {
            return i;
        }
    }
}

int compute_limit(int prediction_minutes) {
    return (prediction_minutes / kiv_ppr_db_connector::MEASURE_INTERVAL_MINUTES) + kiv_ppr_db_connector::COUNT_OF_INPUT_VALUES;
}

kiv_ppr_db_connector::input kiv_ppr_db_connector::load_next(kiv_ppr_db_connector::data_reader* reader, int last_used_first_id, int prediction_minutes) {
    int return_code = 0;
    bool run_again;
    sqlite3* db_handler = reader->db_handler;
    sqlite3_stmt* statement;
    
    std::vector<int> ids;
    std::vector<double> values;
    std::vector<int> segments_id;
    bool is_valid = true;
    const char* sql_command = "select id, ist, segmentid from measuredvalue where id > ? order by id limit ?;";

    int limit = compute_limit(prediction_minutes);

    do {
        ids.clear();
        values.clear();
        segments_id.clear();

        run_again = false;

        if (sqlite3_prepare_v2(db_handler, sql_command, -1, &statement, NULL) != SQLITE_OK) {
            std::cout << "Prepare sql command failure: " << sqlite3_errmsg(db_handler) << std::endl;
            is_valid = false;
            break;
        }

        if (sqlite3_bind_int(statement, 1, last_used_first_id) != SQLITE_OK) {
            std::cout << "Bind parameter 'id' failure: " << sqlite3_errmsg(db_handler) << std::endl;
            is_valid = false;
            break;
        }

        if (sqlite3_bind_int(statement, 2, limit) != SQLITE_OK) {
            std::cout << "Bind parameter 'limit' failure: " << sqlite3_errmsg(db_handler) << std::endl;
            is_valid = false;
            break;
        }

        return_code = sqlite3_step(statement);

        while (return_code == SQLITE_ROW) {
            ids.push_back(sqlite3_column_int(statement, 0));
            values.push_back(sqlite3_column_double(statement, 1));
            segments_id.push_back(sqlite3_column_int(statement, 2));

            return_code = sqlite3_step(statement);
        }

        if (return_code != SQLITE_DONE) {
            std::cout << "SQL execution error: " << sqlite3_errmsg(db_handler) << std::endl;
            is_valid = false;
            break;
        }
        else {
            if (values.size() != (size_t)limit) {
                is_valid = false;
                break;
            }

            if (segments_id[0] != segments_id[segments_id.size() - 1]) {
                last_used_first_id = ids[compute_changed_index(segments_id)];
                sqlite3_finalize(statement);
                run_again = true;
            }

        }

    } while(run_again);

    sqlite3_finalize(statement);

    return create_new_input(ids, values, is_valid);
}
