/**
*
* Konektor do databaze.
*
*/

#pragma once

#include	<vector>
#include	<winsqlite/winsqlite3.h>

namespace kiv_ppr_db_connector {

	/**
	* Struktura pro drzeni inforamci o databazi.
	*
	* db_name - nazev databaze
	* db_hander - handler k databazi
	*/
	struct TData_Reader {
		char *db_name;
		sqlite3 *db_handler;
	};

	/**
	* Struktura pro drzeni pozadovanych dat.
	*
	* ist - hodnota glygemie
	* segment_id - identifikator segmentu
	*/
	struct TElement {
		double ist = 0.0;
		unsigned segment_id = 0;
	};

	/**
	* Vytvori novy reader pro cteni dat z databaze.
	*
	* params:
	*   db_name - nazev databaze
	*
	* return:
	*   novy reader pro cteni dat z databaze
	*/
	TData_Reader New_Reader(char* db_name);

	/**
	* Otevre databazi.
	*
	* params:
	*   reader - handler na databazi
	*
	* return:
	*   true - pokud je otevreni uspesne
	*/
	bool Open_Database(TData_Reader& reader);

	/**
	* Zavre databazi.
	*
	* params:
	*   reader - handler na databazi
	*
	* return:
	*   true - pokud je zavreni uspesne
	*/
	bool Close_Database(TData_Reader& reader);


	/**
	* Nacte data z databaze.
	*
	* params:
	*   reader - handler na databazi
	*
	* return:
	*   vektor dat nactenych z databaze
	*/
	std::vector<TElement> Load_Data(TData_Reader& reader);

}