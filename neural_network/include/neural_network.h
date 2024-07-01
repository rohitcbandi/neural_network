#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "layer.h"
#include "optimizer.h"
#include <vector>
using namespace std;

class NeuralNetwork {
private:
    vector<Layer*> layers;
    Optimizer* optimizer;
    double learningRate;
    int timeStep;

public:
    NeuralNetwork(double learningRate, Optimizer* optimizer = nullptr);
    void addLayer(Layer* layer);
    vector<double> predict(const vector<double>& input);
    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs);
};

#endif // NEURAL_NETWORK_H
