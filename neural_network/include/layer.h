#ifndef LAYER_H
#define LAYER_H

#include <vector>
using namespace std;

class Layer {
public:
    virtual void forward(const vector<double>& input) = 0;
    virtual void backward(const vector<double>& delta) = 0;
    virtual void updateWeights(double learningRate) = 0;
    virtual vector<double> getOutput() = 0;
};

#endif // LAYER_H
