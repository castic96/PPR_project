#include "processor_smp.h"


void Create_Topology(std::vector<unsigned>& topology) {
    topology.clear();

    topology.push_back(INPUT_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN1_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN2_LAYER_NEURONS_COUNT);
    topology.push_back(OUTPUT_LAYER_NEURONS_COUNT);
}

void Create_Neural_Networks(std::vector<kiv_ppr_network::TNetwork>& neural_networks, std::vector<unsigned>& topology) {
    neural_networks.clear();

    for (int i = 0; i < NEURAL_NETWORKS_COUNT; i++) {
        neural_networks.push_back(kiv_ppr_network::New_Network(topology));
    }

}

std::string Generate_Csv(kiv_ppr_network::TNetwork& network) {
    std::string generated_str;
    std::vector<double> relative_errors = network.relative_errors_vector;
    unsigned relative_errors_size = relative_errors.size();
    unsigned step = relative_errors_size / 100;

    generated_str.append(std::to_string(network.average_relative_error)).append(",\n");
    generated_str.append(std::to_string(network.standard_deviation_rel_errs)).append(",\n");

    std::sort(relative_errors.begin(), relative_errors.end());

    for (unsigned i = 0; i < relative_errors_size; i += step) {
        generated_str.append(std::to_string(relative_errors[i]));
        generated_str.append(",");
    }

    return generated_str;
}

std::string Generate_Neural_Ini(kiv_ppr_network::TNetwork& network) {
    std::vector<kiv_ppr_neuron::TLayer> layers = network.layers;
    std::string generated_str;

    for (unsigned i = 1; i < layers.size(); i++) {
        kiv_ppr_neuron::TLayer& current_layer = layers[i];
        unsigned current_neurons_count = current_layer.neurons.size();

        kiv_ppr_neuron::TLayer& previous_layer = layers[i - 1];
        unsigned previous_neurons_count = previous_layer.neurons.size();

        if (i < layers.size() - 1) {
            generated_str.append("[hidden_layer_").append(std::to_string(i)).append("]\n");
        }
        else {
            generated_str.append("[output_layer]\n");
        }

        for (unsigned j = 0; j < current_neurons_count - 1; j++) {

            for (unsigned k = 0; k < previous_neurons_count; k++) {

                if (k < previous_neurons_count - 1) {

                    generated_str.append("Neuron").append(std::to_string(j));
                    generated_str.append("_");
                    generated_str.append("Weight").append(std::to_string(k));
                    generated_str.append("=");
                    generated_str.append(std::to_string(previous_layer.neurons[k].output_weights[j].weight));
                    generated_str.append("\n");
                
                }
                else {

                    generated_str.append("Neuron").append(std::to_string(j));
                    generated_str.append("_");
                    generated_str.append("Bias");
                    generated_str.append("=");
                    generated_str.append(std::to_string(previous_layer.neurons[k].output_weights[j].weight));
                    generated_str.append("\n");

                }

            }

        }

        generated_str.append("\n");
    }

    return generated_str;
}

std::vector<std::vector<double>> Get_Weights(kiv_ppr_network::TNetwork& network) {
    std::vector<std::vector<double>> weights;
    std::vector<kiv_ppr_neuron::TLayer> layers = network.layers;
    std::vector<double> current_weights;

    for (unsigned i = 1; i < layers.size(); i++) {
        kiv_ppr_neuron::TLayer& current_layer = layers[i];
        unsigned current_neurons_count = current_layer.neurons.size();

        kiv_ppr_neuron::TLayer& previous_layer = layers[i - 1];
        unsigned previous_neurons_count = previous_layer.neurons.size();

        current_weights.clear();

        for (unsigned j = 0; j < current_neurons_count - 1; j++) {

            for (unsigned k = 0; k < previous_neurons_count; k++) {

                current_weights.push_back(previous_layer.neurons[k].output_weights[j].weight);
            
            }

        }

        weights.push_back(current_weights);
    }

    return weights;
}

unsigned Find_Best_Network_Index(std::vector<kiv_ppr_network::TNetwork>& neural_networks) {
    unsigned index = 0;
    std::vector<double> total_errors;

    for (unsigned i = 0; i < neural_networks.size(); i++) {
        kiv_ppr_network::TNetwork& current_network = neural_networks[i];

        current_network.average_relative_error = kiv_ppr_utils::Calc_Average_Relative_Error(current_network.relative_errors_vector);
        current_network.standard_deviation_rel_errs = kiv_ppr_utils::Calc_Standard_Deviation(current_network.relative_errors_vector);

        total_errors.push_back(kiv_ppr_utils::Calculate_Total_Error(current_network.relative_errors_vector));
    }

    for (unsigned i = 0; i < total_errors.size(); i++) {
        if (total_errors[i] < total_errors[index]) {
            index = i;
        }
    }

    return index;
}

kiv_ppr_smp::TResults_CPU kiv_ppr_smp::Run_Training_CPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values) {
    kiv_ppr_smp::TResults_CPU result;

    // Vytvoreni topologie
    std::vector<unsigned> topology;
    Create_Topology(topology);

    // Vytvoreni n neuronovych siti s danou topologii
    std::vector<kiv_ppr_network::TNetwork> neural_networks;
    Create_Neural_Networks(neural_networks, topology);

    unsigned num_of_training_sets = expected_values.size();

    for (unsigned i = 0; i < num_of_training_sets; i++) {

        tbb::parallel_for(size_t(0), neural_networks.size(), [&](size_t j) {

            // Spusteni feed forward propagation
            kiv_ppr_network::Feed_Forward_Prop(neural_networks[j], input_values, i);

            // Vypocitani relativni chyby a pridani do vektoru chyb v siti
            kiv_ppr_network::Save_Relative_Error(neural_networks[j], expected_values[i]);

            // Spusteni back propagation
            kiv_ppr_network::Back_Prop(neural_networks[j], target_values, i);

        });

    }

    // Nalezeni nejlepe natrenovane site
    kiv_ppr_network::TNetwork& best_network = neural_networks[Find_Best_Network_Index(neural_networks)];

    result.network = best_network;
    result.relative_errors = best_network.relative_errors_vector;
    result.weights = Get_Weights(best_network);
    result.neural_ini_str = Generate_Neural_Ini(best_network);
    result.csv_str = Generate_Csv(best_network);

    return result;
}