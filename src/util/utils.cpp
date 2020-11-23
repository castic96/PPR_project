#include <random>
#include "utils.h"

double kiv_ppr_utils::Band_Index_To_Level(const size_t index) {
    if (index == 0) {
        return kiv_ppr_utils::Low_Threshold - kiv_ppr_utils::Half_Band_Size;
    }
        
    if (index >= kiv_ppr_utils::Band_Count - 1) {
        return kiv_ppr_utils::High_Threshold + kiv_ppr_utils::Half_Band_Size;
    }
        
    return kiv_ppr_utils::Low_Threshold + static_cast<double>(index - 1)
        * kiv_ppr_utils::Band_Size + kiv_ppr_utils::Half_Band_Size;
}

size_t kiv_ppr_utils::Band_Level_To_Index(double expected_value) {

    // Vraci index prvniho neuronu
    if (expected_value <= kiv_ppr_utils::Low_Threshold) {
        return 0;
    }

    // Vraci index posledniho neuronu
    if (expected_value >= kiv_ppr_utils::High_Threshold) {
        return OUTPUT_LAYER_NEURONS_COUNT - 1;
    }

    int i;

    double interval_ceiling = kiv_ppr_utils::Low_Threshold;

    for (i = 1; i <= kiv_ppr_utils::Internal_Bound_Count; i++) {

        interval_ceiling += kiv_ppr_utils::Band_Size;

        if (interval_ceiling > expected_value) {
            return i;
        }

    }

    return OUTPUT_LAYER_NEURONS_COUNT - 1;
}

double kiv_ppr_utils::Risk_Function(const double bg) {
    // DOI:  10.1080/10273660008833060
    const double original_risk = 1.794 * (pow(log(bg), 1.026) - 1.861);    //mmol/L

    return original_risk / 3.5;
}

std::vector<double> kiv_ppr_utils::Get_Target_Values_Vector(double expected_value) {
    std::vector<double> target_values;
    size_t expected_index;

    target_values.clear();

    // Vytvoreni vektoru o velikosti poctu vystupnich neuronu - inicializace na 0
    for (int i = 0; i < OUTPUT_LAYER_NEURONS_COUNT; i++) {
        target_values.push_back(0);
    }

    expected_index = kiv_ppr_utils::Band_Level_To_Index(expected_value);

    target_values[expected_index] = 1;

    return target_values;
}

double kiv_ppr_utils::Get_Random_Weight() {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_real_distribution<> distribution(0, 1); //uniformni rozdeleni <0,1>

    return distribution(generator);
}

double kiv_ppr_utils::Calc_Average_Relative_Error(std::vector<double> relative_errors_vector) {
    size_t vector_size = relative_errors_vector.size();
    double sum = 0.0;

    for (unsigned i = 0; i < vector_size; i++) {
        sum += relative_errors_vector[i];
    }

    return (sum / (double)vector_size);
}

double kiv_ppr_utils::Calc_Standard_Deviation(std::vector<double> relative_errors_vector) {
    size_t vector_size = relative_errors_vector.size();
    double average_error = kiv_ppr_utils::Calc_Average_Relative_Error(relative_errors_vector);
    double sum = 0.0;

    for (unsigned i = 0; i < vector_size; i++) {
        sum += pow(relative_errors_vector[i] - average_error, 2);
    }

    return (sqrt(sum / (double)vector_size));
}

double kiv_ppr_utils::Calculate_Total_Error(std::vector<double> relative_errors_vector) {
    return kiv_ppr_utils::Calc_Average_Relative_Error(relative_errors_vector) + 
            kiv_ppr_utils::Calc_Standard_Deviation(relative_errors_vector);
}

std::string kiv_ppr_utils::Generate_Csv(std::vector<double> relative_errors_vector) {
    std::string generated_str;
    unsigned relative_errors_size = relative_errors_vector.size();
    unsigned step = relative_errors_size / 100;

    generated_str.append(std::to_string(kiv_ppr_utils::Calc_Average_Relative_Error(relative_errors_vector)));
    generated_str.append(",\n");
    generated_str.append(std::to_string(kiv_ppr_utils::Calc_Standard_Deviation(relative_errors_vector)));
    generated_str.append(",\n");

    std::sort(relative_errors_vector.begin(), relative_errors_vector.end());

    for (unsigned i = 0; i < relative_errors_size; i += step) {
        generated_str.append(std::to_string(relative_errors_vector[i]));
        generated_str.append(",");
    }

    return generated_str;
}

std::string kiv_ppr_utils::Generate_Result_File(std::vector<double>& input_values, std::vector<double>& result_values, std::vector<double>& expected_values, unsigned input_layer_neurons_count) {
    std::string generated_str;
    unsigned train_sets_count = expected_values.size();

    generated_str.append("----------------------------------------------------------------------------------------------");
    generated_str.append("\n");
    generated_str.append("------------------------------------ PREDICTION RESULTS --------------------------------------");
    generated_str.append("\n");
    generated_str.append("----------------------------------------------------------------------------------------------");
    generated_str.append("\n");

    generated_str.append("Number of training sets: ");
    generated_str.append(std::to_string(train_sets_count));
    generated_str.append("\n");

    generated_str.append("----------------------------------------------------------------------------------------------");
    generated_str.append("\n");
    generated_str.append("\n");

    for (unsigned i = 0; i < train_sets_count; i++) {
        generated_str.append("input: ");

        for (unsigned j = 0; j < input_layer_neurons_count; j++) {
            generated_str.append(std::to_string(input_values[j + i * input_layer_neurons_count]));
            generated_str.append("  ");
        }

        generated_str.append("\n");

        generated_str.append("result: ");
        generated_str.append(std::to_string(result_values[i]));
        generated_str.append("\n");

        generated_str.append("expected: ");
        generated_str.append(std::to_string(expected_values[i]));
        generated_str.append("\n\n");   
    }

    return generated_str;
}