#pragma once

#include "neuron.h"

namespace kiv_ppr_layer {

	typedef struct Layer {
		std::vector<kiv_ppr_neuron::neuron> neurons;
	} layer;

	layer new_layer();

}