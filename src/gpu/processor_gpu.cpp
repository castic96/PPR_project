#include "processor_gpu.h"


kiv_ppr_gpu::TResults_GPU kiv_ppr_gpu::Run_Training_GPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values) {
    kiv_ppr_gpu::TResults_GPU result;
    std::vector<double> relative_errors_vector;

    unsigned input_values_size = input_values.size();
    unsigned target_values_size = target_values.size();
    unsigned num_of_training_sets = expected_values.size();

    kiv_ppr_network_gpu::TNetworkGPU network = kiv_ppr_network_gpu::New_Network(input_values, target_values, num_of_training_sets);

    if (!network.is_valid) {
        return result;
    }

    kiv_ppr_network_gpu::Init_Data(network, input_values_size, target_values_size);

    kiv_ppr_network_gpu::Train(network);

    kiv_ppr_network_gpu::Get_Relative_Errors_Vector(network, expected_values, relative_errors_vector);

    kiv_ppr_network_gpu::Clean(network);


    double error = kiv_ppr_utils::Calculate_Total_Error(relative_errors_vector);

    result.network = network;
    result.relative_errors;
    result.weights;

    std::cout << "Chyba: " << error << std::endl;

    return result;
}