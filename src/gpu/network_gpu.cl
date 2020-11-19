// ------ Konstanty -------------------------------------------------------------------------
#define     BIAS							1
#define     INPUT_LAYER_NEURONS_COUNT       8
#define     HIDDEN1_LAYER_NEURONS_COUNT     16
#define     HIDDEN2_LAYER_NEURONS_COUNT     26
#define     OUTPUT_LAYER_NEURONS_COUNT      32
#define     HELPER_DATA_BUFF_SIZE           10
#define		ETA								0.05f
#define		ALPHA							0.1f


// ------ Pomocne funkce pro pristup k datum ------------------------------------------------
// ----- BUFFER: neural_net_data -----
// --- Vrstvy neuronu ---
int input_neuron_i(int i) { 
	return i; 
}

int hidden1_neuron_i(int i) { 
	return input_neuron_i(i) + INPUT_LAYER_NEURONS_COUNT + BIAS; 
}

int hidden2_neuron_i(int i) { 
	return hidden1_neuron_i(i) + HIDDEN1_LAYER_NEURONS_COUNT + BIAS; 
}

int output_neuron_i(int i) { 
	return hidden2_neuron_i(i) + HIDDEN2_LAYER_NEURONS_COUNT + BIAS; 
}

// --- Vahy mezi neurony (bez biasu) ---
int weight_input_hidden1(int input, int hidden1) { 
	return 100 + input * HIDDEN1_LAYER_NEURONS_COUNT + hidden1;  
}

int weight_hidden1_hidden2(int hidden1, int hidden2) {
	return 270 + hidden1 * HIDDEN2_LAYER_NEURONS_COUNT + hidden2; 
}

int weight_hidden2_output(int hidden2, int output) {
	return 750 + hidden2 * OUTPUT_LAYER_NEURONS_COUNT + output;
}

// ----- BUFFER: helper_data -----
// --- Soucet exponencialnich hodnot vsech neuronu vystupni vrstvy (pro SoftMax) ---
int exp_sum_output_layer() {
    return 0;
}

// --- Nejvyssi hodnota z neuronu ve vystupni vrstve ---- 
int max_value_output_layer() {
    return 1;
}


// ----- BUFFER: input_data -----
// --- Pristup k trenovacim datum - vstupy ---
int input_value(int set_num, int n) { 
	return set_num * INPUT_LAYER_NEURONS_COUNT + n; 
}


// ----- BUFFER: target_data -----
// --- Pristup k trenovacim datum - ocekavane vstupy ---
int target_value(int set_num, int n) {
	return set_num * OUTPUT_LAYER_NEURONS_COUNT + n;
}


// ----- BUFFER: result_indexes -----
// --- Pristup k bufferu pro indexy nejvice aktivovanych neuronu vystupni vrstvy ---
int result_value(int set_num) {
	return set_num;
}


// ----- BUFFER: delta_gradient_data -----
// --- Back propagation ---
int delta_input_hidden1(int input, int hidden1) {
	return input * HIDDEN1_LAYER_NEURONS_COUNT + hidden1;
}

int delta_hidden1_hidden2(int hidden1, int hidden2) {
	return 170 + hidden1 * HIDDEN2_LAYER_NEURONS_COUNT + hidden2;
}

int delta_hidden2_output(int hidden2, int output) {
	return 700 + hidden2 * OUTPUT_LAYER_NEURONS_COUNT + output;
}

int	error_gradient_hidden1(int i) {
	return 1700 + i;
}

int	error_gradient_hidden2(int i) {
	return 1750 + i;
}

int error_gradient_output(int i) {
	return 1800 + i;
}


// ------ Pomocne funkce pro vypocty --------------------------------------------------------
float transfer_function_hidden(float value) {
    return tanh(value);
}

float transfer_function_output(float value) {
    return exp(value);
}

float transfer_function_output_der(float value) {
	return 1.0 - (tanh(value) * tanh(value));
}

float transfer_function_hidden_der(float value) {
	return 1.0 - (tanh(value) * tanh(value));
}

inline void atomic_add_float(volatile __global float* source, const float operand) {
	union { unsigned int intVal; float floatVal; } newVal;
	union { unsigned int intVal; float floatVal; } prevVal;
	
    do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int*)source, prevVal.intVal, newVal.intVal ) != prevVal.intVal);
}

 inline void atomic_max_float(volatile __global float *source, const float operand) {
    
    union { unsigned int intVal; float floatVal; } newVal;
    union { unsigned int intVal; float floatVal; } prevVal;
    
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = max(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

float get_output_error_gradient(float target_value, float output_value) {
    return ((target_value - output_value) * transfer_function_output_der(output_value));
}

float get_hidden2_error_gradient(int id, __global float* neural_net_data, __global float* delta_gradient_data) {
    float sum = 0;

    for (int i = 0; i < OUTPUT_LAYER_NEURONS_COUNT; i++) {
        sum += neural_net_data[weight_hidden2_output(id, i)] * delta_gradient_data[error_gradient_output(i)];
    }

    return sum * transfer_function_hidden_der(neural_net_data[hidden2_neuron_i(id)]);
}

float get_hidden1_error_gradient(int id, __global float* neural_net_data, __global float* delta_gradient_data) {
    float sum = 0;

    for (int i = 0; i < HIDDEN2_LAYER_NEURONS_COUNT; i++) {
        sum += neural_net_data[weight_hidden1_hidden2(id, i)] * delta_gradient_data[error_gradient_hidden2(i)];
    }

    return sum * transfer_function_hidden_der(neural_net_data[hidden1_neuron_i(id)]);
}


// ------ Kernely ---------------------------------------------------------------------------
// --- Prirazeni pocatecnich hodnot do vsech vrstev ---
__kernel void set_data_to_layers(__global int* train_set_id, __global float* neural_net_data, __global float* input_data, __global float* helper_data) {
    int id = get_global_id(0);
    
    if (id >= OUTPUT_LAYER_NEURONS_COUNT)
    {
        return;
    }

    neural_net_data[output_neuron_i(id)] = 0;

    if (id < HIDDEN2_LAYER_NEURONS_COUNT) {
        neural_net_data[hidden2_neuron_i(id)] = 0;
	}
    
    if (id < HIDDEN1_LAYER_NEURONS_COUNT) {
        neural_net_data[hidden1_neuron_i(id)] = 0;
	}

    if (id < INPUT_LAYER_NEURONS_COUNT) {
        neural_net_data[input_neuron_i(id)] = input_data[input_value(train_set_id[0], id)];
	}

    if (id < HELPER_DATA_BUFF_SIZE) {
        helper_data[id] = 0;
	}

}

// --- Feed Forward mezi vstupni vrstvou a prvni skrytou vrstvou ---
__kernel void feed_forward_input_hidden1(__global float* neural_net_data) {
    int id = get_global_id(0);
    
    if (id >= HIDDEN1_LAYER_NEURONS_COUNT)
    {
        return;
    }

    for (int i = 0; i <= INPUT_LAYER_NEURONS_COUNT; i++) {
        neural_net_data[hidden1_neuron_i(id)] += 
                            neural_net_data[input_neuron_i(i)] * 
                            neural_net_data[weight_input_hidden1(i, id)];
	}

    neural_net_data[hidden1_neuron_i(id)] = transfer_function_hidden(neural_net_data[hidden1_neuron_i(id)]);

}

// --- Feed Forward mezi prvni a druhou skrytou vrstvou ---
__kernel void feed_forward_hidden1_hidden2(__global float* neural_net_data) {
    int id = get_global_id(0);
    
    if (id >= HIDDEN2_LAYER_NEURONS_COUNT)
    {
        return;
    }

    for (int i = 0; i <= HIDDEN1_LAYER_NEURONS_COUNT; i++) {
        neural_net_data[hidden2_neuron_i(id)] += 
                            neural_net_data[hidden1_neuron_i(i)] * 
                            neural_net_data[weight_hidden1_hidden2(i, id)];
	}

    neural_net_data[hidden2_neuron_i(id)] = transfer_function_hidden(neural_net_data[hidden2_neuron_i(id)]);

}

// --- Feed Forward mezi druhou skrytou vrstvou a vystupni vrstvou ---
__kernel void feed_forward_hidden2_output(__global float* neural_net_data, __global float* helper_data) {
    int id = get_global_id(0);
    
    if (id >= OUTPUT_LAYER_NEURONS_COUNT)
    {
        return;
    }

    for (int i = 0; i <= HIDDEN2_LAYER_NEURONS_COUNT; i++) {
        neural_net_data[output_neuron_i(id)] += 
                            neural_net_data[hidden2_neuron_i(i)] * 
                            neural_net_data[weight_hidden2_output(i, id)];
	}

    neural_net_data[output_neuron_i(id)] = transfer_function_output(neural_net_data[output_neuron_i(id)]);

    // Atomicky soucet - muze pristupovat vice vlaken najednou
    atomic_add_float(&helper_data[exp_sum_output_layer()], neural_net_data[output_neuron_i(id)]);

    // Bariera zastavi vsechna vlakna do te doby, nez vsechna vlakna 
    // prictou do souctu exponencialnich hodnot svoji hodnotu
    barrier(CLK_GLOBAL_MEM_FENCE);

    neural_net_data[output_neuron_i(id)] /= helper_data[exp_sum_output_layer()];

}

// --- Zjisteni indexu neuronu vystupni vrstvy s nejvyssi hodnotou a ulozeni do bufferu ---
__kernel void set_index_of_result(__global int* train_set_id, __global float* neural_net_data, __global int* result_indexes, __global float* helper_data) {
    int id = get_global_id(0);
    
    if (id >= OUTPUT_LAYER_NEURONS_COUNT)
    {
        return;
    }
    
    atomic_max_float(&helper_data[max_value_output_layer()], neural_net_data[output_neuron_i(id)]);

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (isequal(helper_data[max_value_output_layer()], neural_net_data[output_neuron_i(id)])) {
        atomic_xchg(&result_indexes[result_value(train_set_id[0])], id);
	}

}

// --- Back propagation - pro vystupni vrstvu ---
__kernel void back_prop_output(__global int* train_set_id, __global float* neural_net_data, __global float* target_data, __global float* delta_gradient_data) {
    int id = get_global_id(0);
    
    if (id >= OUTPUT_LAYER_NEURONS_COUNT)
    {
        return;
    }

    delta_gradient_data[error_gradient_output(id)] = 
                        get_output_error_gradient(
                            target_data[target_value(train_set_id[0], id)], 
                            neural_net_data[output_neuron_i(id)]
                        );

}

// --- Back propagation - pro druhou skrytou vrstvu ---
__kernel void back_prop_hidden2(__global float* neural_net_data, __global float* delta_gradient_data) {
    int id = get_global_id(0);
    
    if (id > HIDDEN2_LAYER_NEURONS_COUNT)
    {
        return;
    }

    for (int i = 0; i < OUTPUT_LAYER_NEURONS_COUNT; i++) {
        delta_gradient_data[delta_hidden2_output(id, i)] = 
                        ETA * neural_net_data[hidden2_neuron_i(id)] * delta_gradient_data[error_gradient_output(i)] +
                        ALPHA * delta_gradient_data[delta_hidden2_output(id, i)];
	}

    delta_gradient_data[error_gradient_hidden2(id)] = get_hidden2_error_gradient(id, neural_net_data, delta_gradient_data);
}

// --- Back propagation - pro prvni skrytou vrstvu ---
__kernel void back_prop_hidden1(__global float* neural_net_data, __global float* delta_gradient_data) {
    int id = get_global_id(0);
    
    if (id > HIDDEN1_LAYER_NEURONS_COUNT)
    {
        return;
    }

    for (int i = 0; i < HIDDEN2_LAYER_NEURONS_COUNT; i++) {
        delta_gradient_data[delta_hidden1_hidden2(id, i)] = 
                        ETA * neural_net_data[hidden1_neuron_i(id)] * delta_gradient_data[error_gradient_hidden2(i)] +
                        ALPHA * delta_gradient_data[delta_hidden1_hidden2(id, i)];
	}

    delta_gradient_data[error_gradient_hidden1(id)] = get_hidden1_error_gradient(id, neural_net_data, delta_gradient_data);
}

// --- Back propagation - pro vstupni vrstvu ---
__kernel void back_prop_input(__global float* neural_net_data, __global float* delta_gradient_data) {
    int id = get_global_id(0);
    
    if (id > INPUT_LAYER_NEURONS_COUNT)
    {
        return;
    }

    for (int i = 0; i < HIDDEN1_LAYER_NEURONS_COUNT; i++) {
        delta_gradient_data[delta_input_hidden1(id, i)] = 
                        ETA * neural_net_data[input_neuron_i(id)] * delta_gradient_data[error_gradient_hidden1(i)] +
                        ALPHA * delta_gradient_data[delta_input_hidden1(id, i)];
	}
}

// --- Update vah synapsi ---
__kernel void update_weights(__global float* neural_net_data, __global float* delta_gradient_data) {
    int id = get_global_id(0);
    
    if (id >= OUTPUT_LAYER_NEURONS_COUNT)
    {
        return;
    }

    for (int i = 0; i <= HIDDEN2_LAYER_NEURONS_COUNT; i++) {
        neural_net_data[weight_hidden2_output(i, id)] += delta_gradient_data[delta_hidden2_output(i, id)];
	}

    if (id < HIDDEN2_LAYER_NEURONS_COUNT) {

        for (int i = 0; i <= HIDDEN1_LAYER_NEURONS_COUNT; i++ ) {
            neural_net_data[weight_hidden1_hidden2(i, id)] += delta_gradient_data[delta_hidden1_hidden2(i, id)];
		}

	}

    if (id < HIDDEN1_LAYER_NEURONS_COUNT) {

        for (int i = 0; i <= INPUT_LAYER_NEURONS_COUNT; i++) {
            neural_net_data[weight_input_hidden1(i, id)] += delta_gradient_data[delta_input_hidden1(i, id)]; 
		}

	}

}