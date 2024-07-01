#ifndef CONV1D_H
#define CONV1D_H

#include "layer.h"
#include "activation.h"
#include <vector>
using namespace std;

class Conv1D : public Layer {
private:
    int inputSize;
    int filterSize;
    int numFilters;
    Activation activation;
    vector<vector<double>> filters;
    vector<double> biases;
    vector<double> input;
    vector<vector<double>> output;
    vector<vector<double>> filterGradients;
    vector<double> biasGradients;

public:
    Conv1D(int inputSize, int filterSize, int numFilters, Activation activation);
    void forward(const vector<double>& input) override;
    void backward(const vector<double>& delta) override;
    void updateWeights(double learningRate) override;
    vector<double> getOutput() override;
};

#endif // CONV1D_H
