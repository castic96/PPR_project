#include "processor_gpu.h"

//#include "../smp/util/input_parser.h" //pak zmenit minimalne tu cestu aby to nebylo ze smp

int Show_Open_Cl_Info()
{
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
}

void Convert_Values_To_Buff(std::vector<float>& input_values, std::vector<float>& expected_values, float*& input_values_buff, float*& expected_values_buff) {
    std::copy(input_values.begin(), input_values.end(), input_values_buff);
    std::copy(expected_values.begin(), expected_values.end(), expected_values_buff);
}

int Compute_Prediction_Places(unsigned prediction_minutes) {
    return (prediction_minutes / MEASURE_INTERVAL_MINUTES);
}

size_t Compute_Changed_Index(std::vector<kiv_ppr_db_connector::TElement> input_vector, unsigned first_index, int limit) {

    for (unsigned i = first_index; i < first_index + limit - 1; i++) {
        if (input_vector[i].segment_id != input_vector[i + 1].segment_id) {
            return i + 1;
        }
    }

    return 0;
}

void Load_Valid_Inputs(std::vector<kiv_ppr_db_connector::TElement>& input_data,
    std::vector<float>& input_values, std::vector<float>& expected_values, unsigned predicted_minutes) {
    unsigned index = 0;
    bool run_again = false;

    int prediction_places = Compute_Prediction_Places(predicted_minutes);
    int limit = COUNT_OF_INPUT_VALUES + prediction_places;

    while (true) {

        do {
            run_again = false;

            if (index + limit > input_data.size()) {
                return;
            }

            if (input_data[index].segment_id != input_data[index + limit - 1].segment_id) {
                index = Compute_Changed_Index(input_data, index, limit);
                run_again = true;
            }

        } while (run_again);

        for (unsigned i = index; i < index + COUNT_OF_INPUT_VALUES; i++) {
            input_values.push_back((float)input_data[i].ist);
        }

        expected_values.push_back((float)input_data[index + limit - 1].ist);

        index++;
    }

}


/*
void Load_Valid_Inputs(std::vector<kiv_ppr_db_connector::TElement>& input_data, 
    std::vector<float>& input_values, std::vector<float>& expected_values, unsigned predicted_minutes) {
    unsigned index = 0;
    bool run_again = false;
    bool is_first_in_segment = true;
    unsigned counter = 0;

    int prediction_places = Compute_Prediction_Places(predicted_minutes);
    int limit = COUNT_OF_INPUT_VALUES + prediction_places;

    while (true) {

        do {
            run_again = false;

            if (index + limit > input_data.size()) {
                return;
            }

            if (input_data[index].segment_id != input_data[index + limit - 1].segment_id) {

                if (!is_first_in_segment) {

                    for (unsigned i = 0; i < COUNT_OF_INPUT_VALUES - 1; i++) {
                        input_values.push_back((float)input_data[index + i].ist);
                    }

                }

                is_first_in_segment = true;
                index = Compute_Changed_Index(input_data, index, limit);
                run_again = true;

            }

        } while (run_again);

        input_values.push_back((float)input_data[index].ist);
        expected_values.push_back((float)input_data[index + limit - 1].ist);

        is_first_in_segment = false;

        index++;
    }

}
*/

void kiv_ppr_gpu::Run(unsigned predicted_minutes, char*& db_name, char*& weights_file_name) {

    // Otevreni databaze
    kiv_ppr_db_connector::TData_Reader reader = kiv_ppr_db_connector::New_Reader(db_name);

    if (!kiv_ppr_db_connector::Open_Database(reader)) {
        exit(EXIT_FAILURE);
    }

    std::vector<kiv_ppr_db_connector::TElement> input_data = Load_Data(reader);

    kiv_ppr_db_connector::Close_Database(reader);

    std::vector<float> input_values;
    std::vector<float> expected_values;
    Load_Valid_Inputs(input_data, input_values, expected_values, predicted_minutes);

    float *input_values_buff = (float*)calloc(input_values.size(), sizeof(float));
    float *expected_values_buff = (float*)calloc(expected_values.size(), sizeof(float));

    Convert_Values_To_Buff(input_values, expected_values, input_values_buff, expected_values_buff);

    // Zde bych mel mit korektne naplnene buffery









    // Uvolneni pameti bufferu
    free(input_values_buff);
    free(expected_values_buff);

}