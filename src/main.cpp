// BloodGlucoseLevelPredictionChallenge.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "dao/database_connector.h"
#include "neural_network/network.h"
//TODO: pak smazat
#include <cassert>
#include<sstream> 
#include <fstream>
#include <sstream>

#define		START_ID		                0
#define     PREDICTION_MINUTES              60
#define     INPUT_LAYER_NEURONS_COUNT       8
#define     HIDDEN1_LAYER_NEURONS_COUNT     16
#define     HIDDEN2_LAYER_NEURONS_COUNT     26
#define     OUTPUT_LAYER_NEURONS_COUNT      32

using namespace std;

// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned>& topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double>& inputVals);
    unsigned getTargetOutputs(vector<double>& targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned>& topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

void showVectorVals(string label, vector<double>& v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int main()
{
    TrainingData trainData("..\\..\\data\\trainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    kiv_ppr_network::network myNet = kiv_ppr_network::new_network(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        //myNet.feedForward(inputVals);
        kiv_ppr_network::feed_forward_prop(myNet, inputVals);

        // Collect the net's actual output results:
        //myNet.getResults(resultVals);
        kiv_ppr_network::get_results(myNet, resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        //myNet.backProp(targetVals);
        kiv_ppr_network::back_prop(myNet, targetVals);

        // Report how well the training is working, average over recent samples:
        //cout << "Net recent average error: "
        //    << myNet.getRecentAverageError() << endl;

        cout << "Net recent average error: "
            << myNet.recent_average_error << endl;
    }

    cout << endl << "Done" << endl;
}

void exec_data_from_db() {
    kiv_ppr_db_connector::data_reader reader = kiv_ppr_db_connector::new_reader("..\\..\\data\\asc2018.sqlite");

    kiv_ppr_db_connector::open_database(&reader);
    //kiv_ppr_db_connector::load_data(&reader);
    int counter = 0;
    kiv_ppr_db_connector::input current_input = kiv_ppr_db_connector::load_next(&reader, START_ID, PREDICTION_MINUTES);

    while (current_input.valid) {
        counter++;
        std::cout << counter << ": " << std::endl;

        for (size_t i = 0; i < current_input.values.size(); i++) {
            std::cout << "\t" << current_input.values[i] << std::endl;
        }

        std::cout << "\t exp: " << current_input.expected_value << std::endl;

        std::cout << "-------------------------" << std::endl;

        current_input = kiv_ppr_db_connector::load_next(&reader, current_input.first_id, PREDICTION_MINUTES);
    }

    std::cout << "------------ END ----------- " << std::endl;


    kiv_ppr_db_connector::close_database(&reader);
}