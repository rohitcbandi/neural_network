#ifndef RECURRENT_H
#define RECURRENT_H

#include "layer.h"
#include "activation.h"
#include <vector>
using namespace std;

class Recurrent : public Layer {
private:
    int inputSize;
    int hiddenSize;
    Activation activation;
    vector<vector<double>> weights;
    vector<vector<double>> recurrentWeights;
    vector<double> biases;
    vector<double> input;
    vector<double> hiddenState;
    vector<double> output;
    vector<vector<double>> weightGradients;
    vector<vector<double>> recurrentWeightGradients;
    vector<double> biasGradients;

public:
    Recurrent(int inputSize, int hiddenSize, Activation activation);
    void forward(const vector<double>& input) override;
    void backward(const vector<double>& delta) override;
    void updateWeights(double learningRate) override;
    vector<double> getOutput() override;
};

#endif // RECURRENT_H
