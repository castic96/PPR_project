#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 220

#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    <CL/cl2.hpp>
#include    "../constants.h"
#include    "../dao/database_connector.h"

namespace kiv_ppr_gpu {
	void Run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name);
}