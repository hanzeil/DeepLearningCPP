//
// Created by tangjinghao on 17-6-7.
//

#ifndef DEEPLEARNINGCPP_NETWORK_HPP
#define DEEPLEARNINGCPP_NETWORK_HPP

#endif //DEEPLEARNINGCPP_NETWORK_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>
#include <armadillo>

namespace deep_learning_cpp {
    using std::size_t;
    using std::vector;
    using std::cout;
    using std::endl;
    using std::pair;
    using arma::mat;
    using arma::randu;
    using arma::exp;

    class Network {
    public:
        Network(const vector<int> &sizes) : num_layers_(sizes.size()), sizes_(sizes) {
            for (size_t i = 1; i < num_layers_; i++) {
                biases_.push_back(randu < mat > (sizes[i], 1));
                weights_.push_back(randu < mat > (sizes[i], sizes[i - 1]));
            }

        }

    private:
        mat feedforward(mat a) const {
            for (size_t i = 1; i < num_layers_; i++) {
                a = sigmoid(weights_[i] * a + biases_[i]);
            }
            return a;
        }

        void SGD(vector<pair<mat, mat>> &training_data,
                 size_t epochs, size_t mini_batch_size, double eta,
                 const vector<pair<mat, mat>> &test_data) {
            for (size_t i = 0; i < epochs; i++) {
                std::random_shuffle(training_data.begin(), training_data.end());
                for (size_t j = 0; j < training_data.size(); j += mini_batch_size) {
                    update_mini_batch(training_data.begin() + j,
                                      min(training_data.end(), training_data.begin() + j),
                                      eta);
                }
            }
        }

        void update_mini_batch(vector<pair<mat, mat>>::iterator begin,
                               const vector<pair<mat, mat>>::iterator end, double eta) {
            vector<mat> nabla_b(biases_.size(), arma::zeros(biases_[0].n_rows, biases_[0].n_cols));
            vector<mat> nabla_w(biases_.size(), arma::zeros(biases_[0].n_rows, biases_[0].n_cols));
            while (begin < end) {
               
            }
        }

        size_t evaluate(const vector<pair<mat, mat>> &test_data) const {
            size_t result = 0;
            for (const auto &item : test_data) {
                auto max_index = feedforward(item.first).index_max();
                if (item.second[max_index] == 1) result++;
            }
            return result;
        }

        mat sigmoid(const mat &z) const {
            return 1.0 / (1.0 + exp(-z));
        }

        mat sigmoid_prime(const mat &z) const {
            return sigmoid(z) * (1 - sigmoid(z));
        }

        size_t num_layers_;
        const vector<int> sizes_;
        vector<mat> biases_;
        vector<mat> weights_;
    };
}
