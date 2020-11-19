#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <CL/cl2.hpp>
#include    "../dao/database_connector.h"
#include	"network_gpu.h"

namespace kiv_ppr_gpu {
	void Run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name);
}