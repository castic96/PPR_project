#include "layer.h"


kiv_ppr_layer::layer kiv_ppr_layer::new_layer() {
	kiv_ppr_layer::layer new_layer;

	//TODO: možná nebude fungovat..
	new_layer.neurons.clear();

	return new_layer;
}