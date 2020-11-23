#include "processor_gpu.h"

std::string Generate_Label_Neurons(unsigned i, unsigned j) {
    std::string generated_str;

    generated_str.append("Neuron").append(std::to_string(i));
    generated_str.append("_");
    generated_str.append("Weight").append(std::to_string(j));
    generated_str.append("=");

    return generated_str;
}

std::string Generate_Label_Bias(unsigned i) {
    std::string generated_str;

    generated_str.append("Neuron").append(std::to_string(i));
    generated_str.append("_");
    generated_str.append("Bias");
    generated_str.append("=");

    return generated_str;
}

std::string Generate_Neural_Ini_GPU(kiv_ppr_network_gpu::TNetworkGPU& network) {
    std::string generated_str;

    generated_str.append("[hidden_layer_1]\n");

    for (unsigned i = 0; i < HIDDEN1_LAYER_NEURONS_COUNT; i++) {
        for (unsigned j = 0; j < INPUT_LAYER_NEURONS_COUNT + BIAS; j++) {
            
            if (j < INPUT_LAYER_NEURONS_COUNT) {
                generated_str.append(Generate_Label_Neurons(i, j));
            }
            else {
                generated_str.append(Generate_Label_Bias(i));
            }

            generated_str.append(std::to_string(
                network.neural_net_buff[kiv_ppr_mapping_gpu::weight_input_hidden1(j, i)]));

            generated_str.append("\n");

        }
    }

    generated_str.append("\n");
    generated_str.append("[hidden_layer_2]\n");

    for (unsigned i = 0; i < HIDDEN2_LAYER_NEURONS_COUNT; i++) {
        for (unsigned j = 0; j < HIDDEN1_LAYER_NEURONS_COUNT + BIAS; j++) {
            
            if (j < HIDDEN1_LAYER_NEURONS_COUNT) {
                generated_str.append(Generate_Label_Neurons(i, j));
            }
            else {
                generated_str.append(Generate_Label_Bias(i));
            }

            generated_str.append(std::to_string(
                network.neural_net_buff[kiv_ppr_mapping_gpu::weight_hidden1_hidden2(j, i)]));

            generated_str.append("\n");

        }
    }

    generated_str.append("\n");
    generated_str.append("[output_layer]\n");

    for (unsigned i = 0; i < OUTPUT_LAYER_NEURONS_COUNT; i++) {
        for (unsigned j = 0; j < HIDDEN2_LAYER_NEURONS_COUNT + BIAS; j++) {
            
            if (j < HIDDEN2_LAYER_NEURONS_COUNT) {
                generated_str.append(Generate_Label_Neurons(i, j));
            }
            else {
                generated_str.append(Generate_Label_Bias(i));
            }

            generated_str.append(std::to_string(
                network.neural_net_buff[kiv_ppr_mapping_gpu::weight_hidden2_output(j, i)]));

            generated_str.append("\n");

        }
    }

    return generated_str;
}

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


    double error = kiv_ppr_utils::Calculate_Total_Error(relative_errors_vector);

    result.network = network;
    result.relative_errors = relative_errors_vector;
    result.weights;
    result.neural_ini_str = Generate_Neural_Ini_GPU(network);
    result.csv_str = kiv_ppr_utils::Generate_Csv(relative_errors_vector);

    std::cout << "Chyba: " << error << std::endl;

    kiv_ppr_network_gpu::Clean(network);

    return result;
}