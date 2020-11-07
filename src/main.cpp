// BloodGlucoseLevelPredictionChallenge.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include    <iostream>
#include    <stdio.h>
#include    <stdlib.h>
#include    <string>
#include    "dao/database_connector.h"
#include    "neural_network/network.h"

#define		START_ID		                0
#define     PREDICTION_MINUTES              60
#define     INPUT_LAYER_NEURONS_COUNT       8
#define     HIDDEN1_LAYER_NEURONS_COUNT     16
#define     HIDDEN2_LAYER_NEURONS_COUNT     26
#define     OUTPUT_LAYER_NEURONS_COUNT      32
#define     NEURAL_NETWORKS_COUNT           20


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

void run(unsigned& predicted_minutes, char*& db_name, char*& weights_file_name) {

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

    if (weights_file_name != NULL) {

    }
    else {

    }




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