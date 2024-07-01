#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
using namespace std;

class Optimizer {
public:
    virtual void update(vector<double>& params, vector<double>& gradients, int timeStep) = 0;
};

#endif // OPTIMIZER_H
