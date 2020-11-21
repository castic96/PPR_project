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

void kiv_ppr_smp::Run_Training_CPU(std::vector<double>& input_values, std::vector<double>& target_values, std::vector<double>& expected_values) {

    // Vytvoreni topologie
    std::vector<unsigned> topology;
    Create_Topology(topology);

    // Vytvoreni n neuronovych siti s danou topologii
    std::vector<kiv_ppr_network::TNetwork> neural_networks;
    Create_Neural_Networks(neural_networks, topology);

    double relative_error = 0.0;
    std::vector<double> total_errors;

    unsigned num_of_training_sets = expected_values.size();

    for (unsigned i = 0; i < num_of_training_sets; i++) {

        tbb::parallel_for(size_t(0), neural_networks.size(), [&](size_t j) {

            // Spusteni feed forward propagation
            kiv_ppr_network::Feed_Forward_Prop(neural_networks[j], input_values, i);

            // Spusteni back propagation
            kiv_ppr_network::Back_Prop(neural_networks[j], target_values, expected_values[i], i);

            });

    }

    for (unsigned j = 0; j < neural_networks.size(); j++) {
        total_errors.push_back(kiv_ppr_utils::Calculate_Total_Error(neural_networks[j].relative_errors_vector));
    }

    unsigned min_total_error_index = 0;

    for (unsigned j = 0; j < total_errors.size(); j++) {
        if (total_errors[j] < total_errors[min_total_error_index]) {
            min_total_error_index = j;
        }
    }

    // Vypis vsech chyb
    std::cout << "TOTAL ERRORS: ";
    for (unsigned j = 0; j < total_errors.size(); j++) {
        std::cout << total_errors[j] << " ";
    }
    std::cout << std::endl;

    // Vypis chyby pro nejlepsi sit
    std::cout
        << "BEST TOTAL ERROR: "
        << total_errors[min_total_error_index]
        << ", NETWORK INDEX: " << min_total_error_index
        << std::endl;

}