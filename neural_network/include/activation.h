#ifndef ACTIVATION_H
#define ACTIVATION_H

enum Activation {
    RELU,
    SIGMOID,
    TANH
};

double relu(double x);
double relu_derivative(double x);

double sigmoid(double x);
double sigmoid_derivative(double x);

double tanh_(double x);
double tanh_derivative(double x);

using ActivationFunction = double(*)(double);

ActivationFunction getActivationFunction(Activation activation);
ActivationFunction getActivationDerivative(Activation activation);

#endif // ACTIVATION_H
