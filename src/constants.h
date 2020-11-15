#pragma once

#define		START_ID		                0
#define     INPUT_LAYER_NEURONS_COUNT       8
#define     HIDDEN1_LAYER_NEURONS_COUNT     16
#define     HIDDEN2_LAYER_NEURONS_COUNT     26
#define     OUTPUT_LAYER_NEURONS_COUNT      32
#define     NEURAL_NETWORKS_COUNT           20

namespace kiv_ppr_constants {

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

}

