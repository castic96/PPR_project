/**
*
* Konektor do databaze.
*
*/

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <winsqlite/winsqlite3.h>
#include "database_connector.h"


/**
* Vytvori novy reader pro cteni dat z databaze.
*
* params:
*   db_name - nazev databaze
*
* return:
*   novy reader pro cteni dat z databaze
*/
kiv_ppr_db_connector::TData_Reader kiv_ppr_db_connector::New_Reader(char* db_name) {

    kiv_ppr_db_connector::TData_Reader new_reader;

    new_reader.db_name = db_name;

	return new_reader;
}

/**
* Otevre databazi.
*
* params:
*   reader - handler na databazi
*
* return:
*   true - pokud je otevreni uspesne
*/
bool kiv_ppr_db_connector::Open_Database(kiv_ppr_db_connector::TData_Reader &reader) {
    std::cout << "> Opening database '" << reader.db_name << "'..." << std::endl;

    if (sqlite3_open_v2(reader.db_name, &(reader.db_handler), SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
        std::cout << "> Cannot open database: " << sqlite3_errmsg(reader.db_handler) << std::endl;
        return false;
    }

    else
    {
        std::cout << "> Opening database... DONE" << std::endl;
        return true;
    }

}

/**
* Zavre databazi.
*
* params:
*   reader - handler na databazi
*
* return:
*   true - pokud je zavreni uspesne
*/
bool kiv_ppr_db_connector::Close_Database(kiv_ppr_db_connector::TData_Reader& reader) {
    std::cout << "> Closing database..." << std::endl;

    if (sqlite3_close(reader.db_handler) != SQLITE_OK) {
        std::cout << "> Cannot close database: " << sqlite3_errmsg(reader.db_handler) << std::endl;
        return false;
    }

    else
    {
        std::cout << "> Closing database... DONE" << std::endl;
        return true;
    }

}

/**
* Vytvori novy element pro data nactena z databaze.
*
* params:
*   ist - hodnota glygemie
*   segment_id - identifikator segmentu
*
* return:
*   novy element
*/
kiv_ppr_db_connector::TElement New_Element(double ist, unsigned segment_id) {
    kiv_ppr_db_connector::TElement new_element;

    new_element.ist = ist;
    new_element.segment_id = segment_id;

    return new_element;
}

/**
* Nacte data z databaze.
*
* params:
*   reader - handler na databazi
*
* return:
*   vektor dat nactenych z databaze
*/
std::vector<kiv_ppr_db_connector::TElement> kiv_ppr_db_connector::Load_Data(kiv_ppr_db_connector::TData_Reader& reader) {
    std::cout << "> Loading data from database..." << std::endl;

    int return_code = 0;
    std::vector<kiv_ppr_db_connector::TElement> cached_data;
    sqlite3* db_handler = reader.db_handler;
    sqlite3_stmt* statement;

    const char* sql_command = "select ist, segmentid from measuredvalue order by id;";

    if (sqlite3_prepare_v2(db_handler, sql_command, -1, &statement, NULL) != SQLITE_OK) {
        std::cout << "> Prepare sql command failure: " << sqlite3_errmsg(db_handler) << std::endl;
        cached_data.clear();
        return cached_data;
    }

    return_code = sqlite3_step(statement);

    while (return_code == SQLITE_ROW) {

        cached_data.push_back(
            New_Element(
                sqlite3_column_double(statement, 0), 
                sqlite3_column_int(statement, 1)
            )
        );

        return_code = sqlite3_step(statement);
    }


    if (return_code != SQLITE_DONE) {
        std::cout << "> SQL execution error: " << sqlite3_errmsg(db_handler) << std::endl;
        cached_data.clear();
        return cached_data;
    }

    sqlite3_finalize(statement);

    std::cout << "> Loading data from database... DONE" << std::endl;

    return cached_data;
}
