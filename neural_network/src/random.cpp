#include "random.h"
#include <random>

double randomDouble() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-1, 1);
    return dis(gen);
}
