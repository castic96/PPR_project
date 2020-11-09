// BloodGlucoseLevelPredictionChallenge.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include    <iostream>
#include    <stdio.h>
#include    <stdlib.h>
#include    <string>
#include    <cassert>
#include    "constants.h"
#include    "util/mapping.h"
#include    "dao/database_connector.h"
#include    "neural_network/network.h"


void prepare_args(int argc, char** argv, unsigned& predicted_minutes, char*& db_name, char*& weights_file_name) {

    if (argc <= 1) {
        std::cout << "Required arguments have not been entered." << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((argc < 3) || (argc > 4)) {
        std::cout << "Wrong number of arguments have been entered." << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        predicted_minutes = std::stoi(argv[1]);
    }
    catch (...) {
        std::cout << "Wrong type of first argument. Unsigned integer number is required." << std::endl;
        exit(EXIT_FAILURE);
    }

    db_name = argv[2];
    weights_file_name = argv[3];

}

void create_topology(std::vector<unsigned> &topology) {
    topology.clear();

    topology.push_back(INPUT_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN1_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN2_LAYER_NEURONS_COUNT);
    topology.push_back(OUTPUT_LAYER_NEURONS_COUNT);
}

void create_neural_networks(std::vector<kiv_ppr_network::network>& neural_networks, std::vector<unsigned>& topology) {
    neural_networks.clear();

    for (int i = 0; i < NEURAL_NETWORKS_COUNT; i++) {
        neural_networks.push_back(kiv_ppr_network::new_network(topology));
    }

}

std::vector<double> get_target_values_vector(double expected_value) {
    std::vector<double> target_values;
    size_t expected_index;

    target_values.clear();

    // Vytvoreni vektoru o velikosti poctu vystupnich neuronu - inicializace na 0
    for (int i = 0; i < OUTPUT_LAYER_NEURONS_COUNT; i++) {
        target_values.push_back(0);
    }

    expected_index = kiv_ppr_mapping::band_level_to_index(expected_value);

    target_values[expected_index] = 1;

    return target_values;
}

void print_vector(std::string label, std::vector<double>& vector)
{
    std::cout << label << " ";

    for (unsigned i = 0; i < vector.size(); ++i) {
        std::cout << vector[i] << " ";
    }

    std::cout << std::endl;
}

double calculate_relative_error(std::vector<double> result_values, double expected_value) {

    unsigned result_index = 0;
    for (unsigned i = 0; i < result_values.size(); i++) {
        if (result_values[i] > result_values[result_index]) {
            result_index = i;
        }
    }

    double result_value = kiv_ppr_mapping::band_index_to_level(result_index);

    return (std::abs(result_value - expected_value) / expected_value);
}

double calculate_total_error(std::vector<double> relative_errors_vector) {
    size_t vector_size = relative_errors_vector.size();
    double average_error = 0.0;
    double standard_deviation = 0.0;
    double sum = 0.0;

    for (unsigned i = 0; i < vector_size; i++) {
        sum += relative_errors_vector[i];
    }

    average_error = sum / (double)vector_size;

    sum = 0.0;

    for (unsigned i = 0; i < vector_size; i++) {
        sum += pow(relative_errors_vector[i] - average_error, 2);
    }

    standard_deviation = sum / (double)vector_size;

    return average_error + standard_deviation;
}

void run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name) {

    // Otevreni databaze
    kiv_ppr_db_connector::data_reader reader = kiv_ppr_db_connector::new_reader(db_name);
   
    if (!kiv_ppr_db_connector::open_database(&reader)) {
        exit(EXIT_FAILURE);
    }

    // Vytvoreni topologie
    std::vector<unsigned> topology;
    create_topology(topology);

    // Vytvoreni n neuronovych siti s danou topologii
    std::vector<kiv_ppr_network::network> neural_networks;
    create_neural_networks(neural_networks, topology);

    // Nacteni jednoho vstupu
    kiv_ppr_db_connector::input current_input;
    current_input = kiv_ppr_db_connector::load_next(&reader, START_ID, predicted_minutes);

    // Vytvoreni vektoru pro vstupni a ocekavane (cilove) hodnoty
    std::vector<double> input_values;
    std::vector<double> result_values;
    std::vector<double> target_values;

    
    // Zpracovani vsech validnich vstupu
    int counter = 0;
    double relative_error = 0.0;
    double total_error = 0.0;
    std::vector<double> relative_errors_vector;

    while (current_input.valid) {

        // Tisk poradi
        counter++;
        std::cout << counter << ": " << std::endl;

        // Nacteni vstupnich hodnot
        input_values = current_input.values;
        print_vector("Input values:", input_values);

        // Spusteni feed forward propagation
        kiv_ppr_network::feed_forward_prop(neural_networks[0], input_values);

        // Tisk vysledku feed forward propagation
        kiv_ppr_network::get_results(neural_networks[0], result_values);
        print_vector("Result values:", result_values);

        relative_error = calculate_relative_error(result_values, current_input.expected_value);
        relative_errors_vector.push_back(relative_error);
        total_error = calculate_total_error(relative_errors_vector);

        std::cout << "TOTAL ERROR: " << total_error << std::endl;
        
        // TODO: smazat, jen pro test, jestli funguje...
        /*
        double sum = 0.0;
        for (int i = 0; i < result_values.size(); i++) {
            sum += result_values[i];
        }
        std::cout << "SUM: " << sum << std::endl;
        */

        // Nacteni cilovych hodnot
        target_values = get_target_values_vector(current_input.expected_value);
        print_vector("Target values:", target_values);
        std::cout << "Expected value: " << current_input.expected_value << std::endl;

        assert(target_values.size() == topology.back());

        // Spusteni back propagation
        kiv_ppr_network::back_prop(neural_networks[0], target_values);

        std::cout << "Net recent average error: "
            << neural_networks[0].recent_average_error << std::endl;

        std::cout << "----------------------------------------------" << std::endl;

        // Nacteni dalsiho vstupu
        current_input = kiv_ppr_db_connector::load_next(&reader, current_input.first_id, predicted_minutes);
    }




    /*
    if (weights_file_name != NULL) {

    }
    else {

    }
    */




}


int main(int argc, char** argv)
{
    unsigned predicted_minutes;
    char* db_name;
    char* weights_file_name;

    prepare_args(argc, argv, predicted_minutes, db_name, weights_file_name);

    run(predicted_minutes, db_name, weights_file_name);

}

/*
void exec_data_from_db() {
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
}*/