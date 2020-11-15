#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <cassert>
#include    "tbb/parallel_for.h"
#include    "constants.h"
#include    "util/mapping.h"
#include    "dao/database_connector.h"
#include    "neural_network/network.h"
#include    <CL/cl2.hpp>

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

void create_neural_networks(std::vector<kiv_ppr_network::TNetwork>& neural_networks, std::vector<unsigned>& topology) {
    neural_networks.clear();

    for (int i = 0; i < NEURAL_NETWORKS_COUNT; i++) {
        neural_networks.push_back(kiv_ppr_network::New_Network(topology));
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

    expected_index = kiv_ppr_mapping::Band_Level_To_Index(expected_value);

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

    standard_deviation = sqrt(sum / (double)vector_size);

    return average_error + standard_deviation;
}

void run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name) {

    // Otevreni databaze
    kiv_ppr_db_connector::TData_Reader reader = kiv_ppr_db_connector::New_Reader(db_name);
   
    if (!kiv_ppr_db_connector::Open_Database(reader)) {
        exit(EXIT_FAILURE);
    }

    // Vytvoreni topologie
    std::vector<unsigned> topology;
    create_topology(topology);

    // Vytvoreni n neuronovych siti s danou topologii
    std::vector<kiv_ppr_network::TNetwork> neural_networks;
    create_neural_networks(neural_networks, topology);

    // Nacteni jednoho vstupu
    kiv_ppr_db_connector::TInput current_input;
    current_input = kiv_ppr_db_connector::Load_Next(reader, START_ID, predicted_minutes);

    // Vytvoreni vektoru pro vstupni a ocekavane (cilove) hodnoty
    std::vector<double> input_values;
    std::vector<double> result_values;
    std::vector<double> target_values;

    
    // Zpracovani vsech validnich vstupu
    int counter = 0;
    double relative_error = 0.0;
    std::vector<double> total_errors;

    while (current_input.valid) {

        // Tisk poradi
        counter++;

        // Nacteni vstupnich hodnot
        input_values = current_input.values;

        // Nacteni cilovych hodnot
        target_values = get_target_values_vector(current_input.expected_value);

        tbb::parallel_for(size_t(0), neural_networks.size(), [&](size_t i) {
            
            // Spusteni feed forward propagation
            kiv_ppr_network::Feed_Forward_Prop(neural_networks[i], input_values);

            // Spusteni back propagation
            kiv_ppr_network::Back_Prop(neural_networks[i], target_values, current_input.expected_value);

        });

        // Nacteni dalsiho vstupu
        current_input = kiv_ppr_db_connector::Load_Next(reader, current_input.first_id, predicted_minutes);
    }

    for (unsigned i = 0; i < neural_networks.size(); i++) {
        total_errors.push_back(calculate_total_error(neural_networks[i].relative_errors_vector));
    }

    unsigned min_total_error_index = 0;

    for (unsigned i = 0; i < total_errors.size(); i++) {
        if (total_errors[i] < total_errors[min_total_error_index]) {
            min_total_error_index = i;
        }
    }

    // Vypis vsech chyb
    std::cout << "TOTAL ERRORS: ";
    for (unsigned i = 0; i < total_errors.size(); i++) {
        std::cout << total_errors[i] << " ";
    }
    std::cout << std::endl;

    // Vypis chyby pro nejlepsi sit
    std::cout 
        << "BEST TOTAL ERROR: " 
        << total_errors[min_total_error_index]
        << ", NETWORK INDEX: " << min_total_error_index
        << std::endl;

    kiv_ppr_db_connector::Close_Database(reader);


    /*
    if (weights_file_name != NULL) {

    }
    else {

    }
    */

}


int show_open_cl_info()
{
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
}

int main(int argc, char** argv)
{

    show_open_cl_info();

    /*
    unsigned predicted_minutes;
    char* db_name;
    char* weights_file_name;

    prepare_args(argc, argv, predicted_minutes, db_name, weights_file_name);

    run(predicted_minutes, db_name, weights_file_name);
    */

}