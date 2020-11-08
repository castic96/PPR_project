#pragma once

#include "../constants.h"

namespace kiv_ppr_mapping {
	double band_index_to_level(const size_t index);
	size_t band_level_to_index(double expected_value);
}