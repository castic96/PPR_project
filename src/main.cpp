#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    "dao/database_loader.h"
#include    "smp/processor_smp.h"
#include    "gpu/processor_gpu.h"

void Prepare_Args(int argc, char** argv, unsigned& predicted_minutes, char*& db_name, char*& weights_file_name) {

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

void Run(char*& db_name, unsigned& predicted_minutes, char*& weights_file_name, bool run_gpu) {

    // --- Nacteni dat z DB ---
    std::vector<kiv_ppr_db_connector::TElement> input_data = kiv_ppr_database_loader::Load_From_Db(db_name);

    // --- Trenovani site ---
    if (weights_file_name == NULL) {

        std::vector<double> input_values;
        std::vector<double> target_values;
        std::vector<double> expected_values;

        kiv_ppr_database_loader::Load_Inputs_Training(input_data, input_values, target_values, expected_values, predicted_minutes);

        // --- Spusteni na GPU ---
        if (run_gpu) {
            kiv_ppr_gpu::Run_Training_GPU(input_values, target_values, expected_values);
        }

        // --- Spusteni na CPU ---
        else {
            kiv_ppr_smp::Run_Training_CPU(input_values, target_values, expected_values);
        }

    }

    // --- Predikce ---
    else {

        // --- Spusteni na GPU ---
        if (run_gpu) {
        }

        // --- Spusteni na CPU ---
        else {
        }

    }
}

int main(int argc, char** argv)
{

    unsigned predicted_minutes;
    char* db_name;
    char* weights_file_name;
    bool run_gpu = false;

    Prepare_Args(argc, argv, predicted_minutes, db_name, weights_file_name);

    Run(db_name, predicted_minutes, weights_file_name, run_gpu);

}