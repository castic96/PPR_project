#include    <stdio.h>
#include    <stdlib.h>
#include    <iostream>
#include    <string>
#include    "dao/database_connector.h"
#include    "smp/processor_smp.h"
#include    "gpu/processor_gpu.h"

void prepare_args(int argc, char** argv, unsigned& predicted_minutes, char*& db_name, char*& weights_file_name) {

    if (argc <= 1) {
        std::cout << "Required arguments have not been entered." << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((argc < 3) || (argc > 4)) {
        std::cout << "Wrong number of arguments have been entered." << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        predicted_minutes = std::stoi(argv[1]);
    }
    catch (...) {
        std::cout << "Wrong type of first argument. Unsigned integer number is required." << std::endl;
        exit(EXIT_FAILURE);
    }

    db_name = argv[2];
    weights_file_name = argv[3];

}

int main(int argc, char** argv)
{

    //show_open_cl_info();
    unsigned predicted_minutes;
    char* db_name;
    char* weights_file_name;

    prepare_args(argc, argv, predicted_minutes, db_name, weights_file_name);

    //kiv_ppr_smp::Run(predicted_minutes, db_name, weights_file_name);
    kiv_ppr_gpu::Run(predicted_minutes, db_name, weights_file_name);

}