#ifndef DENSE_H
#define DENSE_H

#include "layer.h"
#include "activation.h"
#include <vector>
using namespace std;

class Dense : public Layer {
private:
    int inputSize;
    int outputSize;
    Activation activation;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> input;
    vector<double> output;
    vector<vector<double>> weightGradients;
    vector<double> biasGradients;

public:
    Dense(int inputSize, int outputSize, Activation activation);
    void forward(const vector<double>& input) override;
    void backward(const vector<double>& delta) override;
    void updateWeights(double learningRate) override;
    vector<double> getOutput() override;
};

#endif // DENSE_H
