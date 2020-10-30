// BloodGlucoseLevelPredictionChallenge.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "dao/database_connector.h"

#define		START_ID		0
#define     PREDICTION_MINUTES           60

int main()
{
    //std::cout << "Hello World!\n";
    
    kiv_ppr_db_connector::data_reader reader = kiv_ppr_db_connector::new_reader("..\\..\\data\\asc2018.sqlite");

    kiv_ppr_db_connector::open_database(&reader);
    //kiv_ppr_db_connector::load_data(&reader);
    int counter = 0;
    kiv_ppr_db_connector::input current_input = kiv_ppr_db_connector::load_next(&reader, START_ID, PREDICTION_MINUTES);

    while (current_input.valid) {
        counter++;
        std::cout << counter << ": " << std::endl;

        for (size_t i = 0; i < current_input.values.size(); i++) {
            std::cout << "\t" << current_input.values[i] << std::endl;
        }

        std::cout << "\t exp: " << current_input.expected_value << std::endl;

        std::cout << "-------------------------" << std::endl;

        current_input = kiv_ppr_db_connector::load_next(&reader, current_input.first_id, PREDICTION_MINUTES);
    }

    std::cout << "------------ END ----------- " << std::endl;


    kiv_ppr_db_connector::close_database(&reader);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
