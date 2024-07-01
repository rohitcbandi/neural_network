#include "neural_network.h"
#include <iostream>
using namespace std;

NeuralNetwork::NeuralNetwork(double learningRate, Optimizer* optimizer)
    : learningRate(learningRate), optimizer(optimizer), timeStep(0) {}

void NeuralNetwork::addLayer(Layer* layer) {
    layers.push_back(layer);
}

vector<double> NeuralNetwork::predict(const vector<double>& input) {
    vector<double> currentInput = input;
    for (Layer* layer : layers) {
        layer->forward(currentInput);
        currentInput = layer->getOutput();
    }
    return currentInput;
}

void NeuralNetwork::train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (int i = 0; i < inputs.size(); ++i) {
            vector<double> output = predict(inputs[i]);
            vector<double> delta(output.size());

            for (int j = 0; j < output.size(); ++j) {
                delta[j] = output[j] - targets[i][j];
                loss += delta[j] * delta[j];
            }

            for (int k = layers.size() - 1; k >= 0; --k) {
                layers[k]->backward(delta);
                delta = layers[k]->getOutput();
            }

            timeStep++;
            for (Layer* layer : layers) {
                if (optimizer) {
                    // Update weights using the optimizer
                } else {
                    layer->updateWeights(learningRate);
                }
            }
        }
        loss /= inputs.size();
        cout << "Epoch " << epoch + 1 << " Loss: " << loss << endl;
    }
}
