#include <iostream>
#include <armadillo>
#include "network.hpp"
#include "mnist_loader.hpp"

using namespace std;
using namespace arma;

int main() {

    auto data = deep_learning_cpp::load_data_wrapper();
    vector<int> sizes = {784, 30, 10};
    deep_learning_cpp::Network network(sizes);
    network.SGD(data.first, 200, 10, 3, data.second);
    return 0;
}