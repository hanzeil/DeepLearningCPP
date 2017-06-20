//
// Created by tangjinghao on 17-6-9.
//

#ifndef DEEPLEARNINGCPP_MNIST_LOADER_HPP
#define DEEPLEARNINGCPP_MNIST_LOADER_HPP

#endif //DEEPLEARNINGCPP_MNIST_LOADER_HPP


#include <armadillo>
#include <fstream>
#include <netinet/in.h>

namespace deep_learning_cpp {
    using std::ifstream;
    using std::size_t;
    using std::vector;
    using std::cout;
    using std::endl;
    using std::pair;
    using arma::mat;
    using arma::randu;
    using arma::exp;
    using arma::zeros;

    std::tuple<vector<pair<mat, mat>>, vector<pair<mat, mat>>, vector<pair<mat, mat>>> load_data_wrapper() {
        vector<pair<mat, mat>> training_data(50000, {zeros(784, 1), zeros(10, 1)});
        vector<pair<mat, mat>> validation_data(10000, {zeros(784, 1), zeros(10, 1)});
        vector<pair<mat, mat>> test_data(10000, {zeros(784, 1), zeros(10, 1)});
        ifstream train_inputs_file("../data/train-images-idx3-ubyte", std::ios::binary);
        ifstream train_results_file("../data/train-labels-idx1-ubyte", std::ios::binary);
        ifstream test_inputs_file("../data/t10k-images-idx3-ubyte", std::ios::binary);
        ifstream test_results_file("../data/t10k-labels-idx1-ubyte", std::ios::binary);
        uint32_t len = 0;
        std::vector<char> train_inputs_buffer(
                (std::istreambuf_iterator<char>(train_inputs_file)),
                (std::istreambuf_iterator<char>())
        );
        std::vector<char> train_results_buffer(
                (std::istreambuf_iterator<char>(train_results_file)),
                (std::istreambuf_iterator<char>())
        );
        std::vector<char> test_inputs_buffer(
                (std::istreambuf_iterator<char>(test_inputs_file)),
                (std::istreambuf_iterator<char>())
        );
        std::vector<char> test_results_buffer(
                (std::istreambuf_iterator<char>(test_results_file)),
                (std::istreambuf_iterator<char>())
        );
        train_inputs_file.close();
        train_results_file.close();
        test_inputs_file.close();
        test_results_file.close();
        size_t t1 = 16, t2 = 8;
        for (size_t i = 0; i < 50000; i++) {
            for (size_t j = 0; j < 784; j++) {
                training_data[i].first(j, 0) = ((unsigned char) train_inputs_buffer[t1++]) / 255.0;
            }
            training_data[i].second((size_t) train_results_buffer[t2++], 0) = 1;
        }
        for (size_t i = 0; i < 10000; i++) {
            for (size_t j = 0; j < 784; j++) {
                validation_data[i].first(j, 0) = ((unsigned char) train_inputs_buffer[t1++]) / 255.0;
            }
            validation_data[i].second((size_t) train_results_buffer[t2++], 0) = 1;
        }
        t1 = 16, t2 = 8;
        for (size_t i = 0; i < 10000; i++) {
            for (size_t j = 0; j < 784; j++) {
                test_data[i].first(j, 0) = ((unsigned char) test_inputs_buffer[t1++]) / 255.0;
            }
            test_data[i].second((size_t) test_results_buffer[t2++], 0) = 1;
        }
        return std::make_tuple(training_data, validation_data, test_data);
    };
}
