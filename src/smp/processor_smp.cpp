/**
*
* Hlavni program pro SMP.
*
*/

#include "processor_smp.h"


/**
* Vytvori topologii neuronove site pro trenovani.
*
* params:
*   topology - topologie neuronove site
*/
void Create_Topology_Training(std::vector<unsigned>& topology) {
    topology.clear();

    topology.push_back(INPUT_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN1_LAYER_NEURONS_COUNT);
    topology.push_back(HIDDEN2_LAYER_NEURONS_COUNT);
    topology.push_back(OUTPUT_LAYER_NEURONS_COUNT);
}

/**
* Vytvori topologii neuronove site pro predikci.
*
* params:
*   topology - topologie neuronove site
*   neural_network_params - parametry neuronove site
*/
void Create_Topology_Prediction(std::vector<unsigned>& topology, std::vector<unsigned>& neural_network_params) {
    topology.clear();
    unsigned layers_count = neural_network_params[0];

    for (unsigned i = 0; i < layers_count; i++) {
        topology.push_back(neural_network_params[i + 1]);
    }

}

/**
* Vytvori n neuronovych siti.
*
* params:
*   neural_networks - n neuronovych siti
*   topology - topologie neuronove site
*/
void Create_Neural_Networks(std::vector<kiv_ppr_network::TNetwork>& neural_networks, std::vector<unsigned>& topology) {
    std::cout << "> Creating neural networks..." << std::endl;

    neural_networks.clear();

    for (int i = 0; i < NEURAL_NETWORKS_COUNT; i++) {
        neural_networks.push_back(kiv_ppr_network::New_Network(topology));
    }

    std::cout << "> Creating neural networks... DONE" << std::endl;
    std::cout << "> Count of neural networks: " << neural_networks.size() << std::endl;
}

/**
* Vytvari retezec parametru neuronove site (pro generovani INI souboru).
*
* params:
*   network - neuronova sit
*
* return:
*   retezec parametru neuronove site
*/
std::string Generate_Neural_Ini_CPU(kiv_ppr_network::TNetwork& network) {
    std::cout << "> Generating INI file with parameters of the best trained neural network..." << std::endl;

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

    std::cout << "> Generating INI file with parameters of the best trained neural network... DONE" << std::endl;

    return generated_str;
}

/**
* Ziska vahy synapsi neuronove site.
*
* params:
*   network - neuronova sit
*
* return:
*   vahy neuronove site
*/
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

/**
* Najde index nejlepe natrenovane neuronove site.
*
* params:
*   neural_networks - neuronove site
*
* return:
*   index neuronove site
*/
unsigned Find_Best_Network_Index(std::vector<kiv_ppr_network::TNetwork>& neural_networks) {
    unsigned index = 0;
    std::vector<double> total_errors;

    for (unsigned i = 0; i < neural_networks.size(); i++) {
        kiv_ppr_network::TNetwork& current_network = neural_networks[i];
        total_errors.push_back(kiv_ppr_utils::Calculate_Total_Error(current_network.relative_errors_vector));
    }

    for (unsigned i = 0; i < total_errors.size(); i++) {
        if (total_errors[i] < total_errors[index]) {
            index = i;
        }
    }

    return index;
}

/**
* Nastavi vahy synapsi neuronove site.
*
* params:
*   network - neuronova sit
*   loaded_weights - nactene vahy synapsi neuronove site
*/
void Set_Weights(kiv_ppr_network::TNetwork& network, std::vector<std::vector<double>>& loaded_weights) {
    std::cout << "> Setting loaded weights to neural network..." << std::endl;

    std::vector<kiv_ppr_neuron::TLayer>& layers = network.layers;

    for (unsigned i = 1; i < layers.size(); i++) {
        kiv_ppr_neuron::TLayer& current_layer = layers[i];
        unsigned current_neurons_count = current_layer.neurons.size() - 1;

        kiv_ppr_neuron::TLayer& previous_layer = layers[i - 1];
        unsigned previous_neurons_count = previous_layer.neurons.size();

        for (unsigned j = 0; j < current_neurons_count; j++) {

            for (unsigned k = 0; k < previous_neurons_count; k++) {

                previous_layer.neurons[k].output_weights[j].weight = loaded_weights[i - 1][k + j * previous_neurons_count];
            
            }
        
        }

    }

    std::cout << "> Setting loaded weights to neural network... DONE" << std::endl;

}

/**
* Spousti trenovani neuronove site na SMP.
*
* params:
*   input_values_risk - vektor normalizovanych vstupnich hodnot
*   target_values - vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty
*	expected_values - vektor ocekavanych hodnot
*
* return:
*   vysledky trenovani neuronove site
*/
kiv_ppr_smp::TResults_Training_CPU kiv_ppr_smp::Run_Training_CPU(std::vector<double>& input_values_risk, std::vector<double>& target_values, std::vector<double>& expected_values) {
    kiv_ppr_smp::TResults_Training_CPU result;

    // --- Vytvoreni topologie ---
    std::vector<unsigned> topology;
    Create_Topology_Training(topology);

    // --- Vytvoreni n neuronovych siti s danou topologii ---
    std::vector<kiv_ppr_network::TNetwork> neural_networks;
    Create_Neural_Networks(neural_networks, topology);

    unsigned num_of_training_sets = expected_values.size();

    std::cout << "> Training neural networks..." << std::endl;

    for (unsigned i = 0; i < num_of_training_sets; i++) {

        tbb::parallel_for(size_t(0), neural_networks.size(), [&](size_t j) {

            // --- Spusteni feed forward propagation ---
            kiv_ppr_network::Feed_Forward_Prop(neural_networks[j], input_values_risk, i, topology[0]);

            // --- Vypocitani relativni chyby a pridani do vektoru chyb v siti ---
            kiv_ppr_network::Save_Relative_Error(neural_networks[j], expected_values[i]);

            // --- Spusteni back propagation ---
            kiv_ppr_network::Back_Prop(neural_networks[j], target_values, i, topology[topology.size() - 1]);

            });

    }

    std::cout << "> Training neural networks... DONE" << std::endl;

    // --- Nalezeni nejlepe natrenovane site ---
    kiv_ppr_network::TNetwork& best_network = neural_networks[Find_Best_Network_Index(neural_networks)];

    result.network = best_network;
    result.weights = Get_Weights(best_network);
    result.neural_ini_str = Generate_Neural_Ini_CPU(best_network);
    result.csv_str = kiv_ppr_utils::Generate_Csv(best_network.relative_errors_vector);

    return result;
}

/**
* Spousti predikci na SMP.
*
* params:
*   input_values - vektor vstupnich hodnot
*   input_values_risk - vektor normalizovanych vstupnich hodnot
*	expected_values - vektor ocekavanych hodnot
*	loaded_weights - nactene vahy synapsi pro neuronovou sit
*	neural_network_params - topologie neuronove site (pocet vrstev a neuronu v kazde vrstve)
*
* return:
*   vysledky predikce
*/
kiv_ppr_smp::TResults_Prediction_CPU kiv_ppr_smp::Run_Prediction_CPU(std::vector<double>& input_values,
                                        std::vector<double>& input_values_risk,
                                        std::vector<double>& expected_values,
                                        std::vector<std::vector<double>>& loaded_weights, 
                                        std::vector<unsigned>& neural_network_params) {

    kiv_ppr_smp::TResults_Prediction_CPU result;
    std::vector<double> result_values;

    // --- Vytvoreni topologie ---
    std::vector<unsigned> topology;
    Create_Topology_Prediction(topology, neural_network_params);

    // --- Vytvoreni neuronove site s danou topologii ---
    std::cout << "> Creating neural network..." << std::endl;
    kiv_ppr_network::TNetwork neural_network = kiv_ppr_network::New_Network(topology);
    std::cout << "> Creating neural network... DONE" << std::endl;

    Set_Weights(neural_network, loaded_weights);

    unsigned num_of_training_sets = expected_values.size();

    std::cout << "> Prediction..." << std::endl;

    for (unsigned i = 0; i < num_of_training_sets; i++) {

        // --- Spusteni feed forward propagation ---
        kiv_ppr_network::Feed_Forward_Prop(neural_network, input_values_risk, i, topology[0]);

        // --- Vypocitani relativni chyby a pridani do vektoru chyb v siti ---
        kiv_ppr_network::Save_Relative_Error(neural_network, expected_values[i]);

        // --- Ziskani vysledku predikce ---
        kiv_ppr_network::Add_Result_Value(neural_network, result_values);

    }

    std::cout << "> Prediction... DONE" << std::endl;

    result.network = neural_network;
    result.csv_str = kiv_ppr_utils::Generate_Csv(neural_network.relative_errors_vector);
    result.results = kiv_ppr_utils::Generate_Result_File(input_values, result_values, expected_values, topology[0]);

    return result;
}