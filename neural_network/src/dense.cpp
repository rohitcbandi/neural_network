#include "dense.h"
#include <random>

double randomDouble() {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<> dis(-1, 1);
    return dis(gen);
}

Dense::Dense(int inputSize, int outputSize, Activation activation) 
    : inputSize(inputSize), outputSize(outputSize), activation(activation) {
    weights.resize(outputSize, vector<double>(inputSize));
    biases.resize(outputSize);
    output.resize(outputSize);
    weightGradients.resize(outputSize, vector<double>(inputSize));
    biasGradients.resize(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = randomDouble();
        }
        biases[i] = randomDouble();
    }
}

void Dense::forward(const vector<double>& input) {
    this->input = input;
    ActivationFunction actFunc = getActivationFunction(activation);
    for (int i = 0; i < outputSize; ++i) {
        output[i] = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            output[i] += weights[i][j] * input[j];
        }
        output[i] = actFunc(output[i]);
    }
}

void Dense::backward(const vector<double>& delta) {
    ActivationFunction actDeriv = getActivationDerivative(activation);
    vector<double> newDelta(inputSize, 0.0);

    for (int i = 0; i < outputSize; ++i) {
        double delta_i = delta[i] * actDeriv(output[i]);
        for (int j = 0; j < inputSize; ++j) {
            weightGradients[i][j] += delta_i * input[j];
            newDelta[j] += delta_i * weights[i][j];
        }
        biasGradients[i] += delta_i;
    }
}

void Dense::updateWeights(double learningRate) {
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] -= learningRate * weightGradients[i][j];
            weightGradients[i][j] = 0.0;
        }
        biases[i] -= learningRate * biasGradients[i];
        biasGradients[i] = 0.0;
    }
}

vector<double> Dense::getOutput() {
    return output;
}
