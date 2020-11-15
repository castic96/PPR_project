#pragma once

#include "../constants.h"

namespace kiv_ppr_mapping {
	double Band_Index_To_Level(const size_t index);
	size_t Band_Level_To_Index(double expected_value);
}