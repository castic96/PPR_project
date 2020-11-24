#include "database_loader.h"


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

std::vector<kiv_ppr_db_connector::TElement> kiv_ppr_database_loader::Load_From_Db(char*& db_name) {
    kiv_ppr_db_connector::TData_Reader reader = kiv_ppr_db_connector::New_Reader(db_name);

    if (!kiv_ppr_db_connector::Open_Database(reader)) {
        exit(EXIT_FAILURE);
    }

    std::vector<kiv_ppr_db_connector::TElement> input_data = kiv_ppr_db_connector::Load_Data(reader);

    kiv_ppr_db_connector::Close_Database(reader);

    return input_data;
}

void kiv_ppr_database_loader::Load_Inputs(std::vector<kiv_ppr_db_connector::TElement>& input_data,
    std::vector<double>& input_values,
    std::vector<double>& input_values_risk,
    std::vector<double>& target_values,
    std::vector<double>& expected_values,
    unsigned predicted_minutes, 
    unsigned input_layer_neurons_count) {

    std::vector<double> current_target_values;
    unsigned index = 0;
    bool run_again = false;

    int prediction_places = Compute_Prediction_Places(predicted_minutes);
    int limit = input_layer_neurons_count + prediction_places;

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

        for (unsigned i = index; i < index + input_layer_neurons_count; i++) {
            input_values.push_back(input_data[i].ist);
            input_values_risk.push_back(kiv_ppr_utils::Risk_Function(input_data[i].ist));
        }

        expected_values.push_back(input_data[index + limit - 1].ist);

        current_target_values = kiv_ppr_utils::Get_Target_Values_Vector(input_data[index + limit - 1].ist);

        for (unsigned i = 0; i < current_target_values.size(); i++) {
            target_values.push_back(current_target_values[i]);
        }

        index++;
    }

}