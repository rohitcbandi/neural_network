#include "conv1d.h"
#include "random.h"
#include <random>

Conv1D::Conv1D(int inputSize, int filterSize, int numFilters, Activation activation)
    : inputSize(inputSize), filterSize(filterSize), numFilters(numFilters), activation(activation) {
    filters.resize(numFilters, vector<double>(filterSize));
    biases.resize(numFilters);
    output.resize(numFilters, vector<double>(inputSize - filterSize + 1));
    filterGradients.resize(numFilters, vector<double>(filterSize));
    biasGradients.resize(numFilters);

    for (int i = 0; i < numFilters; ++i) {
        for (int j = 0; j < filterSize; ++j) {
            filters[i][j] = randomDouble();
        }
        biases[i] = randomDouble();
    }
}

void Conv1D::forward(const vector<double>& input) {
    this->input = input;
    ActivationFunction actFunc = getActivationFunction(activation);
    for (int f = 0; f < numFilters; ++f) {
        for (int i = 0; i < inputSize - filterSize + 1; ++i) {
            output[f][i] = biases[f];
            for (int j = 0; j < filterSize; ++j) {
                output[f][i] += filters[f][j] * input[i + j];
            }
            output[f][i] = actFunc(output[f][i]);
        }
    }
}

void Conv1D::backward(const vector<double>& delta) {
    ActivationFunction actDeriv = getActivationDerivative(activation);
    vector<double> newDelta(inputSize, 0.0);

    for (int f = 0; f < numFilters; ++f) {
        for (int i = 0; i < inputSize - filterSize + 1; ++i) {
            double delta_i = delta[i] * actDeriv(output[f][i]);
            for (int j = 0; j < filterSize; ++j) {
                filterGradients[f][j] += delta_i * input[i + j];
                newDelta[i + j] += delta_i * filters[f][j];
            }
            biasGradients[f] += delta_i;
        }
    }
}

void Conv1D::updateWeights(double learningRate) {
    for (int f = 0; f < numFilters; ++f) {
        for (int j = 0; j < filterSize; ++j) {
            filters[f][j] -= learningRate * filterGradients[f][j];
            filterGradients[f][j] = 0.0;
        }
        biases[f] -= learningRate * biasGradients[f];
        biasGradients[f] = 0.0;
    }
}

vector<double> Conv1D::getOutput() {
    vector<double> flattenedOutput;
    for (const auto& filterOutput : output) {
        flattenedOutput.insert(flattenedOutput.end(), filterOutput.begin(), filterOutput.end());
    }
    return flattenedOutput;
}
