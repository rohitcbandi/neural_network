#include "adam.h"
#include <cmath>

Adam::Adam(double learningRate, double beta1, double beta2, double epsilon)
    : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

void Adam::update(vector<double>& params, vector<double>& gradients, int timeStep) {
    if (m.empty()) {
        m.resize(params.size(), 0.0);
        v.resize(params.size(), 0.0);
    }
    for (int i = 0; i < params.size(); ++i) {
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];
        double m_hat = m[i] / (1 - pow(beta1, timeStep));
        double v_hat = v[i] / (1 - pow(beta2, timeStep));
        params[i] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
    }
}
