#include "processor_gpu.h"


kiv_ppr_gpu::TResults_GPU kiv_ppr_gpu::Run_Training_GPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values) {
    kiv_ppr_gpu::TResults_GPU result;
    std::vector<double> relative_errors_vector;
    std::vector<std::vector<double>> relative_errors_all;
    std::vector<double> total_errors;

    unsigned input_values_size = input_values.size();
    unsigned target_values_size = target_values.size();
    unsigned num_of_training_sets = expected_values.size();

    kiv_ppr_network_gpu::TNetworkGPU network = kiv_ppr_network_gpu::New_Network(input_values, target_values, num_of_training_sets);

    if (!network.is_valid) {
        return result;
    }

    for (unsigned i = 0; i < 1; i++) {
        relative_errors_vector.clear();

        kiv_ppr_network_gpu::Init_Data(network, input_values_size, target_values_size);

        kiv_ppr_network_gpu::Train(network);

        kiv_ppr_network_gpu::Get_Relative_Errors_Vector(network, expected_values, relative_errors_vector);

        relative_errors_all.push_back(relative_errors_vector);
    }



    for (unsigned i = 0; i < 1; i++) {
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

    result.network = network;
    result.relative_errors;
    result.weights;

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

    return result;
}