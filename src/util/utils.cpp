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

    /*
    // TODO: mozna predelat
    double band_size_modified = 
        (kiv_ppr_constants::High_Threshold - kiv_ppr_constants::Low_Threshold) 
        / static_cast<double>(kiv_ppr_constants::Internal_Bound_Count - 2);

    double interval_ceiling = kiv_ppr_constants::Low_Threshold;

    for (i = 1; i < kiv_ppr_constants::Internal_Bound_Count - 1; i++) {

        interval_ceiling += band_size_modified;

        if (interval_ceiling > expected_value) {
            break;
        }

    }*/


    // TODO: mozna predelat
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