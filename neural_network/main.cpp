#include "neural_network.h"
#include "dense.h"
#include "adam.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
    NeuralNetwork nn(0.01, new Adam());

    nn.addLayer(new Dense(2, 3, RELU));
    nn.addLayer(new Dense(3, 1, SIGMOID));

    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    nn.train(inputs, targets, 1000);

    for (const auto& input : inputs) {
        vector<double> output = nn.predict(input);
        cout << "Input: (" << input[0] << ", " << input[1] << ") -> Output: " << output[0] << endl;
    }

    return 0;
}
