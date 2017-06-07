#include <iostream>
#include <armadillo>
#include "network.hpp"

using namespace std;
using namespace arma;

int main() {
    vector<int> sizes = {784, 30, 10};
    deep_learning_cpp::Network network(sizes);
    return 0;
}