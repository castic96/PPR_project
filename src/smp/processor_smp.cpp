#include "processor_smp.h"

void Create_Topology(std::vector<unsigned>& topology) {
    topology.clear();

    topology.push_back(INPUT_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN1_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN2_LAYER_NEURONS_COUNT);
    topology.push_back(OUTPUT_LAYER_NEURONS_COUNT);
}

void Create_Neural_Networks(std::vector<kiv_ppr_network::TNetwork>& neural_networks, std::vector<unsigned>& topology) {
    neural_networks.clear();

    for (int i = 0; i < NEURAL_NETWORKS_COUNT; i++) {
        neural_networks.push_back(kiv_ppr_network::New_Network(topology));
    }

}

void Print_Vector(std::string label, std::vector<double>& vector)
{
    std::cout << label << " ";

    for (unsigned i = 0; i < vector.size(); ++i) {
        std::cout << vector[i] << " ";
    }

    std::cout << std::endl;
}

double Calculate_Total_Error(std::vector<double> relative_errors_vector) {
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

void kiv_ppr_smp::Run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name) {

    // Otevreni databaze
    kiv_ppr_db_connector::TData_Reader reader = kiv_ppr_db_connector::New_Reader(db_name);

    if (!kiv_ppr_db_connector::Open_Database(reader)) {
        exit(EXIT_FAILURE);
    }

    // Vytvoreni topologie
    std::vector<unsigned> topology;
    Create_Topology(topology);

    // Vytvoreni n neuronovych siti s danou topologii
    std::vector<kiv_ppr_network::TNetwork> neural_networks;
    Create_Neural_Networks(neural_networks, topology);

    // Nacteni jednoho vstupu
    std::vector<kiv_ppr_db_connector::TElement> input_data = Load_Data(reader);
    kiv_ppr_db_connector::Close_Database(reader);

    kiv_ppr_input_parser::TInput current_input;
    current_input = kiv_ppr_input_parser::Read_Next(input_data, START_ID, predicted_minutes);

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
        target_values = kiv_ppr_utils::Get_Target_Values_Vector(current_input.expected_value);

        tbb::parallel_for(size_t(0), neural_networks.size(), [&](size_t i) {

            // Spusteni feed forward propagation
            kiv_ppr_network::Feed_Forward_Prop(neural_networks[i], input_values);

            // Spusteni back propagation
            kiv_ppr_network::Back_Prop(neural_networks[i], target_values, current_input.expected_value);

            });

        // Nacteni dalsiho vstupu
        current_input = kiv_ppr_input_parser::Read_Next(input_data, current_input.first_index, predicted_minutes);
    }

    for (unsigned i = 0; i < neural_networks.size(); i++) {
        total_errors.push_back(Calculate_Total_Error(neural_networks[i].relative_errors_vector));
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


    /*
    if (weights_file_name != NULL) {

    }
    else {

    }
    */

}