// ------ Konstanty -------------------------------------------------------------------------
//
// --- Neuron neprezentujici bias ---
#define     BIAS							1

// --- Pocet neuronu vstupni vrstvy ---
#define     INPUT_LAYER_NEURONS_COUNT       8

// --- Pocet neuronu prvni skryte vrstvy ---
#define     HIDDEN1_LAYER_NEURONS_COUNT     16

// --- Pocet neuronu druhe skryte vrstvy ---
#define     HIDDEN2_LAYER_NEURONS_COUNT     26

// --- Pocet neuronu vystupni vrstvy ---
#define     OUTPUT_LAYER_NEURONS_COUNT      32

// --- Velikost pomocneho bufferu ---
#define     HELPER_DATA_BUFF_SIZE           10

// --- Rychlost uceni site [0.0 - 1.0] ---
#define		ETA								0.01f

// --- Multiplikator posledni zmeny vahy (momentum) [0.0 - n] ---
#define		ALPHA							0.1f


/**
* Mapuje neuron vstupni vrstvy v bufferu 'neural_net_data'.
*
* params:
*   i - index neuronu
*
* return:
*   namapovana hodnota
*/
int input_neuron_i(int i) { 
	return i; 
}

/**
* Mapuje neuron prvni skryte vrstvy v bufferu 'neural_net_data'.
*
* params:
*   i - index neuronu
*
* return:
*	namapovana hodnota
*/
int hidden1_neuron_i(int i) { 
	return input_neuron_i(i) + INPUT_LAYER_NEURONS_COUNT + BIAS; 
}

/**
* Mapuje neuron druhe skryte vrstvy v bufferu 'neural_net_data'.
*
* params:
*   i - index neuronu
*
* return:
*	namapovana hodnota
*/
int hidden2_neuron_i(int i) { 
	return hidden1_neuron_i(i) + HIDDEN1_LAYER_NEURONS_COUNT + BIAS; 
}

/**
* Mapuje neuron vystupni vrstvy v bufferu 'neural_net_data'.
*
* params:
*   i - index neuronu
*
* return:
*	namapovana hodnota
*/
int output_neuron_i(int i) { 
	return hidden2_neuron_i(i) + HIDDEN2_LAYER_NEURONS_COUNT + BIAS; 
}

/**
* Mapuje vahy neuronu mezi vstupni a prvni skrytou vrstvou v bufferu 'neural_net_data'.
*
* params:
*   input - index neuronu ve vstupni vrstve
*	hidden1 - index neuronu v prvni skryte vrstve
*
* return:
*	namapovana hodnota
*/
int weight_input_hidden1(int input, int hidden1) { 
	return 100 + input * HIDDEN1_LAYER_NEURONS_COUNT + hidden1;  
}

/**
* Mapuje vahy neuronu mezi prvni skrytou a druhou skrytou vrstvou v bufferu 'neural_net_data'.
*
* params:
*   hidden1 - index neuronu v prvni skryte vrstve
*	hidden2 - index neuronu ve druhe skryte vrstve
*
* return:
*	namapovana hodnota
*/
int weight_hidden1_hidden2(int hidden1, int hidden2) {
	return 270 + hidden1 * HIDDEN2_LAYER_NEURONS_COUNT + hidden2; 
}

/**
* Mapuje vahy neuronu mezi druhou skrytou a vystupni vrstvou v bufferu 'neural_net_data'.
*
* params:
*   hidden2 - index neuronu ve druhe skryte vrstve
*	output - index neuronu ve vystupni vrstve
*
* return:
*	namapovana hodnota
*/
int weight_hidden2_output(int hidden2, int output) {
	return 750 + hidden2 * OUTPUT_LAYER_NEURONS_COUNT + output;
}

/**
* Vraci hodnotu indexu v bufferu 'helper_data', kde se nachazi soucet exponencialnich 
* hodnot vsech neuronu vystupni vrstvy (pro SoftMax).
*
* return:
*	hodnota indexu
*/
int exp_sum_output_layer() {
    return 0;
}

/**
* Vraci hodnotu indexu v bufferu 'helper_data', kde se nachazi nejvyssi 
* hodnota neuronu vystupni vrstvy.
*
* return:
*	hodnota indexu
*/
int max_value_output_layer() {
    return 1;
}

/**
* Mapuje index neuronu vstupni vrstvy na vstupni hodnoty v bufferu 'input_data'.
*
* params:
*	set_num - index trenovaciho vzorku
*   n - index neuronu
*
* return:
*	namapovana hodnota
*/
int input_value(int set_num, int n) { 
	return set_num * INPUT_LAYER_NEURONS_COUNT + n; 
}

/**
* Mapuje index neuronu vystupni vrstvy na vystupnich hodnoty v bufferu 'target_data'.
*
* params:
*	set_num - index trenovaciho vzorku
*   n - index neuronu
*
* return:
*	namapovana hodnota
*/
int target_value(int set_num, int n) {
	return set_num * OUTPUT_LAYER_NEURONS_COUNT + n;
}

/**
* Mapuje index trenovaciho vzorku na buffer 'result_indexes', kde jsou ulozeny indexy
* nejvice aktivovanych neuronu vystupni vrstvy.
*
* params:
*	set_num - index trenovaciho vzorku
*
* return:
*	namapovana hodnota
*/
int result_value(int set_num) {
	return set_num;
}

/**
* Mapuje delty vah mezi vstupni a prvni skrytou vrstvou v bufferu 'delta_gradient_data'.
*
* params:
*   input - index neuronu ve druhe skryte vrstve
*	hidden1 - index neuronu ve vystupni vrstve
*
* return:
*	namapovana hodnota
*/
int delta_input_hidden1(int input, int hidden1) {
	return input * HIDDEN1_LAYER_NEURONS_COUNT + hidden1;
}

/**
* Mapuje delty vah mezi prvni skrytou a druhou skrytou vrstvou v bufferu 'delta_gradient_data'.
*
* params:
*   hidden1 - index neuronu ve druhe skryte vrstve
*	hidden2 - index neuronu ve vystupni vrstve
*
* return:
*	namapovana hodnota
*/
int delta_hidden1_hidden2(int hidden1, int hidden2) {
	return 170 + hidden1 * HIDDEN2_LAYER_NEURONS_COUNT + hidden2;
}

/**
* Mapuje delty vah mezi druhou skrytou a vystupni skrytou vrstvou v bufferu 'delta_gradient_data'.
*
* params:
*   hidden2 - index neuronu ve druhe skryte vrstve
*	output - index neuronu ve vystupni vrstve
*
* return:
*	namapovana hodnota
*/
int delta_hidden2_output(int hidden2, int output) {
	return 700 + hidden2 * OUTPUT_LAYER_NEURONS_COUNT + output;
}

/**
* Mapuje gradienty pro prvni skrytou vrstvu v bufferu 'delta_gradient_data'.
*
* params:
*   i - index neuronu
*
* return:
*	namapovana hodnota
*/
int	error_gradient_hidden1(int i) {
	return 1700 + i;
}

/**
* Mapuje gradienty pro druhou skrytou vrstvu v bufferu 'delta_gradient_data'.
*
* params:
*   i - index neuronu
*
* return:
*	namapovana hodnota
*/
int	error_gradient_hidden2(int i) {
	return 1750 + i;
}

/**
* Mapuje gradienty pro vystupni vrstvu v bufferu 'delta_gradient_data'.
*
* params:
*   i - index neuronu
*
* return:
*	namapovana hodnota
*/
int error_gradient_output(int i) {
	return 1800 + i;
}


/**
* Aktivacni funkce pro skryte vrstvy.
*
* params:
*   value - vstupni hodnota aktivacni funkce
*
* return:
*	vystupni hodnota aktivacni funkce
*/
float transfer_function_hidden(float value) {
    return tanh(value);
}

/**
* Aktivacni funkce pro vystupni vrstvu.
*
* params:
*   value - vstupni hodnota aktivacni funkce
*
* return:
*	vystupni hodnota aktivacni funkce
*/
float transfer_function_output(float value) {
    return exp(value);
}

/**
* Derivace aktivacni funkce pro skryte vrstvy.
*
* params:
*   value - vstupni hodnota derivace aktivacni funkce
*
* return:
*	vystupni hodnota derivace aktivacni funkce
*/
float transfer_function_hidden_der(float value) {
	return 1.0 - (tanh(value) * tanh(value));
}

/**
* Derivace aktivacni funkce pro vystupni vrstvu.
*
* params:
*   value - vstupni hodnota derivace aktivacni funkce
*
* return:
*	vystupni hodnota derivace aktivacni funkce
*/
float transfer_function_output_der(float value) {
	return 1.0 - (tanh(value) * tanh(value));
}

/**
* Atomicky soucet float hodnot.
*
* params:
*   source - hodnota 1 (k teto hodnote se pricita)
*   operand - hodnota 2 (tato hodnota se pricita)
*/
inline void atomic_add_float(volatile __global float* source, const float operand) {
	union { unsigned int intVal; float floatVal; } newVal;
	union { unsigned int intVal; float floatVal; } prevVal;
	
    do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int*)source, prevVal.intVal, newVal.intVal ) != prevVal.intVal);
}

/**
* Atomicky max float hodnot.
*
* params:
*   source - hodnota 1 (tato hodnota je nahrazovana)
*   operand - hodnota 2 (touto hodnotou se nahrazuje)
*/
 inline void atomic_max_float(volatile __global float *source, const float operand) {
    
    union { unsigned int intVal; float floatVal; } newVal;
    union { unsigned int intVal; float floatVal; } prevVal;
    
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = max(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

/**
* Vypocita gradient vystupni vrstvy.
*
* params:
*   target_value - ocekavana hodnota
*	output_value - vystupni hodnota
*
* return:
*   hodnota gradientu
*/
float get_output_error_gradient(float target_value, float output_value) {
    return ((target_value - output_value) * transfer_function_output_der(output_value));
}

/**
* Vypocita gradient druhe skryte vrstvy.
*
* params:
*   id - index neuronu
*	neural_net_data - data neuronove site
*	delta_gradient_data - delty a gradienty
*
* return:
*   hodnota gradientu
*/
float get_hidden2_error_gradient(int id, __global float* neural_net_data, __global float* delta_gradient_data) {
    float sum = 0;

    for (int i = 0; i < OUTPUT_LAYER_NEURONS_COUNT; i++) {
        sum += neural_net_data[weight_hidden2_output(id, i)] * delta_gradient_data[error_gradient_output(i)];
    }

    return sum * transfer_function_hidden_der(neural_net_data[hidden2_neuron_i(id)]);
}

/**
* Vypocita gradient prvni skryte vrstvy.
*
* params:
*   id - index neuronu
*	neural_net_data - buffer pro ulozeni hodnot neuronu a vah 
*	delta_gradient_data - delty a gradienty neuronove site
*
* return:
*   hodnota gradientu
*/
float get_hidden1_error_gradient(int id, __global float* neural_net_data, __global float* delta_gradient_data) {
    float sum = 0;

    for (int i = 0; i < HIDDEN2_LAYER_NEURONS_COUNT; i++) {
        sum += neural_net_data[weight_hidden1_hidden2(id, i)] * delta_gradient_data[error_gradient_hidden2(i)];
    }

    return sum * transfer_function_hidden_der(neural_net_data[hidden1_neuron_i(id)]);
}


/**
* Kernel pro zvyseni pocitadla trenovaci mnoziny.
*
* params:
*   train_set_id - index trenovaciho vzorku
*/
__kernel void inc_train_set_id(__global int* train_set_id) {
    int id = get_global_id(0);

    if (id > 1)
    {
        return;
    }

    train_set_id[0]++;
}

/**
* Kernel pro prirazeni pocatecnich hodnot do vsech vrstev.
*
* params:
*   train_set_id - index trenovaciho vzorku
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah 
*   input_data - vstupni hodnoty trenovaci mnoziny
*   helper_data - buffer pro pomocne vypocty na zarizeni
*/
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

/**
* Kernel pro feed forward mezi vstupni vrstvou a prvni skrytou vrstvou.
*
* params:
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah
*/
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

/**
* Kernel pro feed forward mezi prvni a druhou skrytou vrstvou.
*
* params:
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah
*/
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

/**
* Kernel pro feed forward mezi druhou skrytou vrstvou a vystupni vrstvou.
*
* params:
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah
*   helper_data - buffer pro pomocne vypocty na zarizeni
*/
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

    neural_net_data[output_neuron_i(id)] = transfer_function_hidden(neural_net_data[output_neuron_i(id)]);

}

/**
* Kernel pro zjisteni indexu neuronu vystupni vrstvy s nejvyssi hodnotou a ulozeni do bufferu.
*
* params:
*   train_set_id - index trenovaciho vzorku
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah 
*   result_indexes - buffer pro ulozeni indexu neuronu vystupni vrstvy s nejvyssi hodnotou
*   helper_data - buffer pro pomocne vypocty na zarizeni
*/
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

/**
* Kernel pro back propagation - pro vystupni vrstvu.
*
* params:
*   train_set_id - index trenovaciho vzorku
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah 
*   target_data - ocekavane hodnoty trenovaci mnoziny
*   delta_gradient_data - delty a gradienty neuronove site
*/
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

/**
* Kernel pro back propagation - pro druhou skrytou vrstvu.
*
* params:
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah
*   delta_gradient_data - delty a gradienty neuronove site
*/
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

/**
* Kernel pro back propagation - pro prvni skrytou vrstvu.
*
* params:
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah
*   delta_gradient_data - delty a gradienty neuronove site
*/
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

/**
* Kernel pro back propagation - pro vstupni vrstvu.
*
* params:
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah
*   delta_gradient_data - delty a gradienty neuronove site
*/
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

/**
* Kernel pro update vah synapsi.
*
* params:
*   neural_net_data - buffer pro ulozeni hodnot neuronu a vah
*   delta_gradient_data - delty a gradienty neuronove site
*/
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