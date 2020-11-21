#include "processor_gpu.h"


int Compute_Prediction_Places(unsigned prediction_minutes) {
    return (prediction_minutes / MEASURE_INTERVAL_MINUTES);
}

size_t Compute_Changed_Index(std::vector<kiv_ppr_db_connector::TElement> input_vector, unsigned first_index, int limit) {

    for (unsigned i = first_index; i < first_index + limit - 1; i++) {
        if (input_vector[i].segment_id != input_vector[i + 1].segment_id) {
            return i + 1;
        }
    }

    return 0;
}

void Load_Valid_Inputs(std::vector<kiv_ppr_db_connector::TElement>& input_data,
    std::vector<cl_float>& input_values, std::vector<cl_float>& target_values, std::vector<double>& expected_values, unsigned predicted_minutes) {
    std::vector<double> current_target_values;
    unsigned index = 0;
    bool run_again = false;

    int prediction_places = Compute_Prediction_Places(predicted_minutes);
    int limit = COUNT_OF_INPUT_VALUES + prediction_places;

    while (true) {

        do {
            run_again = false;

            if (index + limit > input_data.size()) {
                return;
            }

            if (input_data[index].segment_id != input_data[index + limit - 1].segment_id) {
                index = Compute_Changed_Index(input_data, index, limit);
                run_again = true;
            }

        } while (run_again);

        for (unsigned i = index; i < index + COUNT_OF_INPUT_VALUES; i++) {
            input_values.push_back((cl_float)kiv_ppr_utils::Risk_Function(input_data[i].ist));
        }

        expected_values.push_back(input_data[index + limit - 1].ist);

        current_target_values = kiv_ppr_utils::Get_Target_Values_Vector(input_data[index + limit - 1].ist);

        for (unsigned i = 0; i < current_target_values.size(); i++) {
            target_values.push_back((cl_float)current_target_values[i]);
        }

        index++;
    }

}


/*
void Load_Valid_Inputs(std::vector<kiv_ppr_db_connector::TElement>& input_data, 
    std::vector<float>& input_values, std::vector<float>& expected_values, unsigned predicted_minutes) {
    unsigned index = 0;
    bool run_again = false;
    bool is_first_in_segment = true;
    unsigned counter = 0;

    int prediction_places = Compute_Prediction_Places(predicted_minutes);
    int limit = COUNT_OF_INPUT_VALUES + prediction_places;

    while (true) {

        do {
            run_again = false;

            if (index + limit > input_data.size()) {
                return;
            }

            if (input_data[index].segment_id != input_data[index + limit - 1].segment_id) {

                if (!is_first_in_segment) {

                    for (unsigned i = 0; i < COUNT_OF_INPUT_VALUES - 1; i++) {
                        input_values.push_back((float)input_data[index + i].ist);
                    }

                }

                is_first_in_segment = true;
                index = Compute_Changed_Index(input_data, index, limit);
                run_again = true;

            }

        } while (run_again);

        input_values.push_back((float)input_data[index].ist);
        expected_values.push_back((float)input_data[index + limit - 1].ist);

        is_first_in_segment = false;

        index++;
    }

}
*/

std::vector<kiv_ppr_db_connector::TElement> Load_From_Db(char*& db_name) {
    kiv_ppr_db_connector::TData_Reader reader = kiv_ppr_db_connector::New_Reader(db_name);

    if (!kiv_ppr_db_connector::Open_Database(reader)) {
        exit(EXIT_FAILURE);
    }

    std::vector<kiv_ppr_db_connector::TElement> input_data = kiv_ppr_db_connector::Load_Data(reader);

    kiv_ppr_db_connector::Close_Database(reader);

    return input_data;
}

void kiv_ppr_gpu::Run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name) {
    std::vector<double> relative_errors_vector;
    std::vector<std::vector<double>> relative_errors_all;
    std::vector<double> total_errors;

    // Nacteni dat z db
    std::vector<kiv_ppr_db_connector::TElement> input_data = Load_From_Db(db_name);

    // Vytvoreni vektoru vstupu a ocekavanych hodnot
    std::vector<cl_float> input_values;
    std::vector<cl_float> target_values;
    std::vector<double> expected_values;

    Load_Valid_Inputs(input_data, input_values, target_values, expected_values, predicted_minutes);

    unsigned input_values_size = input_values.size();
    unsigned target_values_size = target_values.size();
    unsigned num_of_training_sets = expected_values.size();

    kiv_ppr_network_gpu::TNetworkGPU network = kiv_ppr_network_gpu::New_Network(input_values, target_values, num_of_training_sets);

    if (!network.is_valid) {
        return;
    }

    for (unsigned i = 0; i < 20; i++) {
        relative_errors_vector.clear();

        kiv_ppr_network_gpu::Init_Data(network, input_values_size, target_values_size);

        kiv_ppr_network_gpu::Train(network);

        kiv_ppr_network_gpu::Get_Relative_Errors_Vector(network, expected_values, relative_errors_vector);

        relative_errors_all.push_back(relative_errors_vector);
    }



    for (unsigned i = 0; i < 20; i++) {
        total_errors.push_back(kiv_ppr_utils::Calculate_Total_Error(relative_errors_all[i]));
        
    }

    //double error = kiv_ppr_utils::Calculate_Total_Error(relative_errors_vector);

    //TODO: nekde tady uvolnit buffery

    //TODO: pak smazat
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

    //std::cout << "Chyba: " << error << std::endl;
}