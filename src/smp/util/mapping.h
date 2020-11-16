#pragma once

#include "../../constants.h"

namespace kiv_ppr_mapping {

    //mmol/L below which a medical attention is needed
    static constexpr double Low_Threshold = 3.0;

    //dtto above
    static constexpr double High_Threshold = 13.0;

    //number of bounds inside the thresholds
    static constexpr size_t Internal_Bound_Count = OUTPUT_LAYER_NEURONS_COUNT - 2;

    //must imply relative error <= 10%
    static constexpr double Band_Size =
        (High_Threshold - Low_Threshold) / static_cast<double>(Internal_Bound_Count);

    //abs(Low_Threshold-Band_Size)/Low_Threshold
    static constexpr double Inv_Band_Size = 1.0 / Band_Size;

    static constexpr double Half_Band_Size = 0.5 / Inv_Band_Size;

    static constexpr size_t Band_Count = Internal_Bound_Count + 2;

	
    double Band_Index_To_Level(const size_t index);
	size_t Band_Level_To_Index(double expected_value);
}