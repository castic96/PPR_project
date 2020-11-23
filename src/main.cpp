#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    "dao/database_loader.h"
#include    "smp/processor_smp.h"
#include    "gpu/processor_gpu.h"
#include    "util/svg_generator.h"
#include    "dao/file_manager.h"
#include    "constants.h"

#define     CSV_PATH                "../out/errors.csv"
#define     NEURAL_INI_PATH         "../out/neural.ini"
#define     RESULTS_PATH            "../out/results.txt"
#define     GREEN_GRAPH_PATH        "../out/green_graph.svg"
#define     BLUE_GRAPH_PATH         "../out/blue_graph.svg"


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
    
    // --- Promenne pro vstupy ---
    std::vector<double> input_values;
    std::vector<double> input_values_risk;
    std::vector<double> target_values;
    std::vector<double> expected_values;

    // --- Promenne pro vytupy --- 
    std::string green_graph;
    std::string blue_graph;
    std::string neural_ini_str;
    std::string csv_str;
    std::string results_str;

    // --- Nacteni dat z DB ---
    std::vector<kiv_ppr_db_connector::TElement> input_data = kiv_ppr_database_loader::Load_From_Db(db_name);

    // --- Trenovani site ---
    if (weights_file_name == NULL) {
        kiv_ppr_database_loader::Load_Inputs_Training(input_data, input_values, 
                                        input_values_risk, target_values, 
                                        expected_values, predicted_minutes, 
                                        INPUT_LAYER_NEURONS_COUNT);

        // --- Spusteni na GPU ---
        if (run_gpu) {
            kiv_ppr_gpu::TResults_Training_GPU result_training_gpu = kiv_ppr_gpu::Run_Training_GPU(input_values_risk, target_values, expected_values);
            
            neural_ini_str = result_training_gpu.neural_ini_str;
            csv_str = result_training_gpu.csv_str;
        }

        // --- Spusteni na CPU ---
        else {

            kiv_ppr_smp::TResults_Training_CPU result_training_cpu = kiv_ppr_smp::Run_Training_CPU(input_values_risk, target_values, expected_values);

            neural_ini_str = result_training_cpu.neural_ini_str;
            csv_str = result_training_cpu.csv_str;

            kiv_ppr_svg_generator::TSvg_Generator svg_generator = kiv_ppr_svg_generator::New_Generator(result_training_cpu.network);
            kiv_ppr_svg_generator::Generate(svg_generator, green_graph, blue_graph);

            kiv_ppr_file_manager::Save_Svg_File(GREEN_GRAPH_PATH, green_graph);
            kiv_ppr_file_manager::Save_Svg_File(BLUE_GRAPH_PATH, blue_graph);
        }

    kiv_ppr_file_manager::Save_Ini_File(NEURAL_INI_PATH, neural_ini_str);
    kiv_ppr_file_manager::Save_Csv_File(CSV_PATH, csv_str);

    }

    // --- Predikce ---
    else {
        std::vector<std::vector<double>> loaded_weights;
        std::vector<unsigned> neural_network_params;
        kiv_ppr_file_manager::Load_Ini_File(weights_file_name, loaded_weights, neural_network_params);

        kiv_ppr_database_loader::Load_Inputs_Training(input_data, input_values,
            input_values_risk, target_values,
            expected_values, predicted_minutes,
            neural_network_params[1]);

        // --- Spusteni na GPU ---
        if (run_gpu) {
        }

        // --- Spusteni na CPU ---
        else {
            kiv_ppr_smp::TResults_Prediction_CPU result_prediction_cpu = kiv_ppr_smp::Run_Prediction_CPU(
                input_values, input_values_risk, expected_values, loaded_weights, neural_network_params);

            csv_str = result_prediction_cpu.csv_str;
            results_str = result_prediction_cpu.results;

            kiv_ppr_svg_generator::TSvg_Generator svg_generator = kiv_ppr_svg_generator::New_Generator(result_prediction_cpu.network);
            kiv_ppr_svg_generator::Generate(svg_generator, green_graph, blue_graph);

            kiv_ppr_file_manager::Save_Svg_File(GREEN_GRAPH_PATH, green_graph);
            kiv_ppr_file_manager::Save_Svg_File(BLUE_GRAPH_PATH, blue_graph);
        }

        kiv_ppr_file_manager::Save_Csv_File(CSV_PATH, csv_str);
        kiv_ppr_file_manager::Save_Results_File(RESULTS_PATH, results_str);

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