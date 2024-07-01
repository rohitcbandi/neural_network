#include "recurrent.h"
#include "random.h"
#include <random>

Recurrent::Recurrent(int inputSize, int hiddenSize, Activation activation)
    : inputSize(inputSize), hiddenSize(hiddenSize), activation(activation) {
    weights.resize(hiddenSize, vector<double>(inputSize));
    recurrentWeights.resize(hiddenSize, vector<double>(hiddenSize));
    biases.resize(hiddenSize);
    hiddenState.resize(hiddenSize);
    output.resize(hiddenSize);
    weightGradients.resize(hiddenSize, vector<double>(inputSize));
    recurrentWeightGradients.resize(hiddenSize, vector<double>(hiddenSize));
    biasGradients.resize(hiddenSize);

    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = randomDouble();
        }
        for (int k = 0; k < hiddenSize; ++k) {
            recurrentWeights[i][k] = randomDouble();
        }
        biases[i] = randomDouble();
    }
}

void Recurrent::forward(const vector<double>& input) {
    this->input = input;
    ActivationFunction actFunc = getActivationFunction(activation);
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenState[i] = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            hiddenState[i] += weights[i][j] * input[j];
        }
        for (int k = 0; k < hiddenSize; ++k) {
            hiddenState[i] += recurrentWeights[i][k] * hiddenState[k];
        }
        hiddenState[i] = actFunc(hiddenState[i]);
    }
    output = hiddenState;
}

void Recurrent::backward(const vector<double>& delta) {
    ActivationFunction actDeriv = getActivationDerivative(activation);
    vector<double> newDelta(inputSize, 0.0);

    for (int i = 0; i < hiddenSize; ++i) {
        double delta_i = delta[i] * actDeriv(output[i]);
        for (int j = 0; j < inputSize; ++j) {
            weightGradients[i][j] += delta_i * input[j];
            newDelta[j] += delta_i * weights[i][j];
        }
        for (int k = 0; k < hiddenSize; ++k) {
            recurrentWeightGradients[i][k] += delta_i * hiddenState[k];
        }
        biasGradients[i] += delta_i;
    }
}

void Recurrent::updateWeights(double learningRate) {
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] -= learningRate * weightGradients[i][j];
            weightGradients[i][j] = 0.0;
        }
        for (int k = 0; k < hiddenSize; ++k) {
            recurrentWeights[i][k] -= learningRate * recurrentWeightGradients[i][k];
            recurrentWeightGradients[i][k] = 0.0;
        }
        biases[i] -= learningRate * biasGradients[i];
        biasGradients[i] = 0.0;
    }
}

vector<double> Recurrent::getOutput() {
    return output;
}
