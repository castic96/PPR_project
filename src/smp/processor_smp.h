#pragma once

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <cassert>
#include    "tbb/parallel_for.h"
#include    "../constants.h"
#include    "../util/utils.h"
#include    "../dao/database_connector.h"
#include	"../util/input_parser.h"
#include    "neural_network/network.h"

namespace kiv_ppr_smp {

	void Run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name);

}