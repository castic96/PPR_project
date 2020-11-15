#include "mapping.h"

double kiv_ppr_mapping::Band_Index_To_Level(const size_t index) {
    if (index == 0) {
        return kiv_ppr_constants::Low_Threshold - kiv_ppr_constants::Half_Band_Size;
    }
        
    if (index >= kiv_ppr_constants::Band_Count - 1) {
        return kiv_ppr_constants::High_Threshold + kiv_ppr_constants::Half_Band_Size;
    }
        
    return kiv_ppr_constants::Low_Threshold + static_cast<double>(index - 1) 
        * kiv_ppr_constants::Band_Size + kiv_ppr_constants::Half_Band_Size;
}

size_t kiv_ppr_mapping::Band_Level_To_Index(double expected_value) {

    // Vraci index prvniho neuronu
    if (expected_value <= kiv_ppr_constants::Low_Threshold) {
        return 0;
    }

    // Vraci index posledniho neuronu
    if (expected_value >= kiv_ppr_constants::High_Threshold) {
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
    double interval_ceiling = kiv_ppr_constants::Low_Threshold;

    for (i = 1; i <= kiv_ppr_constants::Internal_Bound_Count; i++) {

        interval_ceiling += kiv_ppr_constants::Band_Size;

        if (interval_ceiling > expected_value) {
            return i;
        }

    }

    return OUTPUT_LAYER_NEURONS_COUNT - 1;
}