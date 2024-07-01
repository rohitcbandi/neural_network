#include "activation.h"
#include <cmath>

double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1 : 0; }

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

double tanh_(double x) { return tanh(x); }
double tanh_derivative(double x) { return 1 - tanh(x) * tanh(x); }

ActivationFunction getActivationFunction(Activation activation) {
    switch (activation) {
        case RELU: return relu;
        case SIGMOID: return sigmoid;
        case TANH: return tanh_;
    }
    return nullptr;
}

ActivationFunction getActivationDerivative(Activation activation) {
    switch (activation) {
        case RELU: return relu_derivative;
        case SIGMOID: return sigmoid_derivative;
        case TANH: return tanh_derivative;
    }
    return nullptr;
}
