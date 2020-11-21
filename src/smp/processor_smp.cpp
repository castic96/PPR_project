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

int Compute_Prediction_Places1(unsigned prediction_minutes) {
    return (prediction_minutes / MEASURE_INTERVAL_MINUTES);
}

size_t Compute_Changed_Index1(std::vector<kiv_ppr_db_connector::TElement> input_vector, unsigned first_index, int limit) {

    for (unsigned i = first_index; i < first_index + limit - 1; i++) {
        if (input_vector[i].segment_id != input_vector[i + 1].segment_id) {
            return i + 1;
        }
    }

    return 0;
}


void Load_Valid_Inputs1(std::vector<kiv_ppr_db_connector::TElement>& input_data,
    std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values, unsigned predicted_minutes) {
    std::vector<double> current_target_values;
    unsigned index = 0;
    bool run_again = false;

    int prediction_places = Compute_Prediction_Places1(predicted_minutes);
    int limit = COUNT_OF_INPUT_VALUES + prediction_places;

    while (true) {

        do {
            run_again = false;

            if (index + limit > input_data.size()) {
                return;
            }

            if (input_data[index].segment_id != input_data[index + limit - 1].segment_id) {
                index = Compute_Changed_Index1(input_data, index, limit);
                run_again = true;
            }

        } while (run_again);

        for (unsigned i = index; i < index + COUNT_OF_INPUT_VALUES; i++) {
            input_values.push_back(kiv_ppr_utils::Risk_Function(input_data[i].ist));
        }

        expected_values.push_back(input_data[index + limit - 1].ist);

        current_target_values = kiv_ppr_utils::Get_Target_Values_Vector(input_data[index + limit - 1].ist);

        for (unsigned i = 0; i < current_target_values.size(); i++) {
            target_values.push_back(current_target_values[i]);
        }

        index++;
    }

}

std::vector<kiv_ppr_db_connector::TElement> Load_From_Db1(char*& db_name) {
    kiv_ppr_db_connector::TData_Reader reader = kiv_ppr_db_connector::New_Reader(db_name);

    if (!kiv_ppr_db_connector::Open_Database(reader)) {
        exit(EXIT_FAILURE);
    }

    std::vector<kiv_ppr_db_connector::TElement> input_data = kiv_ppr_db_connector::Load_Data(reader);

    kiv_ppr_db_connector::Close_Database(reader);

    return input_data;
}


void kiv_ppr_smp::Run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name) {

    // Otevreni databaze
    kiv_ppr_db_connector::TData_Reader reader = kiv_ppr_db_connector::New_Reader(db_name);

    if (!kiv_ppr_db_connector::Open_Database(reader)) {
        exit(EXIT_FAILURE);
    }


    std::vector<kiv_ppr_db_connector::TElement> input_data = Load_From_Db1(db_name);
    // Vytvoreni vektoru vstupu a ocekavanych hodnot
    std::vector<double> input_values;
    std::vector<double> target_values;
    std::vector<double> expected_values;

    Load_Valid_Inputs1(input_data, input_values, target_values, expected_values, predicted_minutes);

    unsigned num_of_training_sets = expected_values.size();

    // Vytvoreni topologie
    std::vector<unsigned> topology;
    Create_Topology(topology);

    // Vytvoreni n neuronovych siti s danou topologii
    std::vector<kiv_ppr_network::TNetwork> neural_networks;
    Create_Neural_Networks(neural_networks, topology);
    /*
    // Nacteni jednoho vstupu
    std::vector<kiv_ppr_db_connector::TElement> input_data = Load_Data(reader);
    kiv_ppr_db_connector::Close_Database(reader);

    kiv_ppr_input_parser::TInput current_input;
    current_input = kiv_ppr_input_parser::Read_Next(input_data, START_ID, predicted_minutes);

    // Vytvoreni vektoru pro vstupni a ocekavane (cilove) hodnoty
    std::vector<double> input_values;
    std::vector<double> result_values;
    std::vector<double> target_values;*/


    // Zpracovani vsech validnich vstupu
    int counter = 0;
    double relative_error = 0.0;
    std::vector<double> total_errors;

    for (unsigned j = 0; j < num_of_training_sets; j++) {

        // Tisk poradi
        counter++;

        // Nacteni vstupnich hodnot
        //input_values = current_input.values;

        // Nacteni cilovych hodnot
        //target_values = kiv_ppr_utils::Get_Target_Values_Vector(current_input.expected_value);

        tbb::parallel_for(size_t(0), neural_networks.size(), [&](size_t i) {

            // Spusteni feed forward propagation
            kiv_ppr_network::Feed_Forward_Prop(neural_networks[i], input_values, j);

            // Spusteni back propagation
            kiv_ppr_network::Back_Prop(neural_networks[i], target_values, expected_values[j], j);

            });

        // Nacteni dalsiho vstupu
        //current_input = kiv_ppr_input_parser::Read_Next(input_data, current_input.first_index, predicted_minutes);
    }

    for (unsigned i = 0; i < neural_networks.size(); i++) {
        total_errors.push_back(kiv_ppr_utils::Calculate_Total_Error(neural_networks[i].relative_errors_vector));
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