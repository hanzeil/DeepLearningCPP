#include <iostream>
#include <armadillo>
#include "network2.hpp"
#include "mnist_loader.hpp"

using namespace std;
using namespace arma;

int main() {

    vector<pair<mat, mat>> training_data, validation_data, test_data;
    std::tie(training_data, validation_data, test_data) = deep_learning_cpp::load_data_wrapper();
    vector<int> sizes = {784, 30, 10};
    deep_learning_cpp::network2::Network2 network(sizes);
    network.save("model.bin");
    auto model = deep_learning_cpp::network2::load("model.bin");
    model.SGD(training_data, 300, 10, 0.5, 5.0, validation_data, true, true, true, true);

    return 0;
}