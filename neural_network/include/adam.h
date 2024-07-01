#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"
#include <vector>
using namespace std;

class Adam : public Optimizer {
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    vector<double> m;
    vector<double> v;

public:
    Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
    void update(vector<double>& params, vector<double>& gradients, int timeStep) override;
};

#endif // ADAM_H
