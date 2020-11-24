/**
*
* Hlavni soubor programu obsahujici spousteci bod.
*
*/

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


/**
* Vypise do konzole informace o programu.
*/
void Print_Info() {
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "-----------------------  BLOOD GLUCOSE LEVEL PREDICTION CHALLENGE 1.0 --------------------" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "--  KIV/PPR - Semestral project  ---------------------------------------------------------" << std::endl;
    std::cout << "--  2020/2021  ---------------------------------------------------------------------------" << std::endl;
    std::cout << "--  Zdenek Castoral, A19N0026P  ----------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

/**
* Vypise do konzole napovedu ke spusteni programu.
*/
void Print_Help() {
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "  USAGE:" << std::endl;
    std::cout << "    Program arguments:" << std::endl << std::endl;
    std::cout << "\t\t<prediction_minutes> <database> [device] [neural_params]" << std::endl << std::endl;
    std::cout << "\t- prediction_minutes: minutes to predict - must be unsigned int divisible by 5" << std::endl;
    std::cout << "\t- database: name of the database to load training set" << std::endl;
    std::cout << "\t- device (optional): CPU/GPU - 0=CPU (default), 1=GPU" << std::endl;
    std::cout << "\t- neural_params (optional): file with the weights of the neural network" << std::endl;
    std::cout << std::endl;
    std::cout << "    If the 'neural_params' file is entered, the type of device cannot be entered - CPU\n    will be used." << std::endl;

    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

/**
* Vypise do konzole informace o aktualni konfiguraci programu.
*
* params:
*   db_name - nazev databaze k nacteni dat
*   predicted_minutes - pocet minut dopredu, na ktery se bude predikovat nebo trenovat
*   weights_file_name - soubor s vahami neuronove site
*   run_gpu - logicka hodnota, zda ma program trenovat na GPU
*/
void Print_Conf(char*& db_name, unsigned& predicted_minutes, char*& weights_file_name, bool run_gpu) {
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "  RUN CONFIGURATION:" << std::endl;
    std::cout << "\tGeneral:" << std::endl;
    std::cout << "\t\tScale:\t\t\trisk function" << std::endl;
    std::cout << "\t\tClassification:\t\tmulticlass" << std::endl;
    std::cout << "\tChosen:" << std::endl;
    std::cout << "\t\tOperation:\t\t" << ((weights_file_name == NULL) ? "training" : "prediction") << std::endl;
    std::cout << "\t\tDevice:\t\t\t" << ((run_gpu) ? "GPU" : "CPU") << std::endl;
    std::cout << "\t\tMinutes:\t\t" << predicted_minutes << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

/**
* Parsuje argumenty prikazove radky.
*
* params:
*   argc - pocet argumentu prikazove radky
*   argv - argumenty prikazove radky
*   predicted_minutes - pocet minut dopredu, na ktery se bude predikovat nebo trenovat
*   db_name - nazev databaze k nacteni dat
*   run_gpu - logicka hodnota, zda ma program trenovat na GPU
*   weights_file_name - soubor s vahami neuronove site
*/
void Prepare_Args(int argc, char** argv, unsigned& predicted_minutes, char*& db_name, bool& run_gpu, char*& weights_file_name) {

    if (argc <= 1) {
        std::cout << "> Required arguments have not been entered." << std::endl;
        Print_Help();
        exit(EXIT_FAILURE);
    }

    if ((argc < 3) || (argc > 4)) {
        std::cout << "> Wrong number of arguments have been entered." << std::endl;
        Print_Help();
        exit(EXIT_FAILURE);
    }

    try {
        predicted_minutes = std::stoi(argv[1]);
    }
    catch (...) {
        std::cout << "> Wrong type of first argument. Unsigned integer number is required." << std::endl;
        Print_Help();
        exit(EXIT_FAILURE);
    }

    if (predicted_minutes <= 0 || predicted_minutes % 5 != 0) {
        std::cout << "> Wrong type of first argument. Integer must by unsigned and divisible by 5." << std::endl;
        Print_Help();
        exit(EXIT_FAILURE);
    }

    db_name = argv[2];
    run_gpu = false;
    weights_file_name = NULL;
    
    if (argc == 4) {

        if (strlen(argv[3]) == 1 && (*argv[3] == '1' || *argv[3] == '0')) {
            
            if (*argv[3] == '1') {
                run_gpu = true;
            }

        }
        else {
            weights_file_name = argv[3];
        }
    }

}

/**
* Beh programu. Zakladni vetveni, zda se pouzije CPU/GPU a zda se bude
* predikovat nebo trenovat.
*
* params:
*   db_name - nazev databaze k nacteni dat
*   predicted_minutes - pocet minut dopredu, na ktery se bude predikovat nebo trenovat
*   weights_file_name - soubor s vahami neuronove site
*   run_gpu - logicka hodnota, zda ma program trenovat na GPU
*/
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
        kiv_ppr_database_loader::Load_Inputs(input_data, input_values,
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

        if (!kiv_ppr_file_manager::Load_Ini_File(weights_file_name, loaded_weights, neural_network_params)) {
            exit(EXIT_FAILURE);
        }

        kiv_ppr_database_loader::Load_Inputs(input_data, input_values,
            input_values_risk, target_values,
            expected_values, predicted_minutes,
            neural_network_params[1]);

        kiv_ppr_smp::TResults_Prediction_CPU result_prediction_cpu = kiv_ppr_smp::Run_Prediction_CPU(
            input_values, input_values_risk, expected_values, loaded_weights, neural_network_params);

        csv_str = result_prediction_cpu.csv_str;
        results_str = result_prediction_cpu.results;

        kiv_ppr_svg_generator::TSvg_Generator svg_generator = kiv_ppr_svg_generator::New_Generator(result_prediction_cpu.network);
        kiv_ppr_svg_generator::Generate(svg_generator, green_graph, blue_graph);

        kiv_ppr_file_manager::Save_Svg_File(GREEN_GRAPH_PATH, green_graph);
        kiv_ppr_file_manager::Save_Svg_File(BLUE_GRAPH_PATH, blue_graph);
        kiv_ppr_file_manager::Save_Csv_File(CSV_PATH, csv_str);
        kiv_ppr_file_manager::Save_Results_File(RESULTS_PATH, results_str);
    }
}

/**
* Spousteci bod programu.
*
* params:
*   argc - pocet argumentu prikazove radky
*   argv - argumenty prikazove radky
*
* return:
*   navratovy kod programu   
*/
int main(int argc, char** argv) {

    unsigned predicted_minutes;
    char* db_name;
    char* weights_file_name;
    bool run_gpu;

    Print_Info();

    Prepare_Args(argc, argv, predicted_minutes, db_name, run_gpu, weights_file_name);

    Print_Conf(db_name, predicted_minutes, weights_file_name, run_gpu);

    Run(db_name, predicted_minutes, weights_file_name, run_gpu);

}