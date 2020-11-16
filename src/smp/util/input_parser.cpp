#include "input_parser.h"


kiv_ppr_input_parser::TInput New_Input(std::vector<double> values, double expected_value, unsigned first_index, bool is_valid) {
    kiv_ppr_input_parser::TInput new_input;
    
    new_input.values = values;
    new_input.expected_value = expected_value;
    new_input.first_index = first_index;
    new_input.valid = is_valid;

    return new_input;
}

size_t Compute_Changed_Index(std::vector<kiv_ppr_db_connector::TElement> input_vector, unsigned first_index, int limit) {

    for (unsigned i = first_index; i < first_index + limit - 1; i++) {
        if (input_vector[i].segment_id != input_vector[i + 1].segment_id) {
            return i + 1;
        }
    }

    return 0;
}

int Compute_Prediction_Places(unsigned prediction_minutes) {
    return (prediction_minutes / kiv_ppr_input_parser::MEASURE_INTERVAL_MINUTES);
}

kiv_ppr_input_parser::TInput kiv_ppr_input_parser::Read_Next(std::vector<kiv_ppr_db_connector::TElement> input_vector, int last_used_first_index, unsigned prediction_minutes) {
    std::vector<double> values;
    double expected_value = 0.0;
    bool run_again = false;
    unsigned first_index = last_used_first_index + 1;
    
    int prediction_places = Compute_Prediction_Places(prediction_minutes);
    int limit = kiv_ppr_input_parser::COUNT_OF_INPUT_VALUES + prediction_places;

    do {

        run_again = false;

        if (first_index + limit > input_vector.size()) {
            return New_Input(values, expected_value, first_index, false);
        }

        if (input_vector[first_index].segment_id != input_vector[first_index + limit - 1].segment_id) {
            first_index = Compute_Changed_Index(input_vector, first_index, limit);
            run_again = true;
        }

    } while (run_again);

    for (unsigned i = first_index; i < first_index + kiv_ppr_input_parser::COUNT_OF_INPUT_VALUES; i++) {
        values.push_back(input_vector[i].ist);
    }

    expected_value = input_vector[first_index + limit - 1].ist;

    return New_Input(values, expected_value, first_index, true);
}