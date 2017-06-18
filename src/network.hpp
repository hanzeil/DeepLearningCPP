//
// Created by tangjinghao on 17-6-7.
//

#ifndef DEEPLEARNINGCPP_NETWORK_HPP
#define DEEPLEARNINGCPP_NETWORK_HPP

#endif //DEEPLEARNINGCPP_NETWORK_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <armadillo>

namespace deep_learning_cpp {
    using std::size_t;
    using std::vector;
    using std::cout;
    using std::endl;
    using std::pair;
    using std::ifstream;
    using arma::mat;
    using arma::randn;
    using arma::zeros;
    using arma::exp;

    class Network {
    public:
        Network(const vector<int> &sizes) : num_layers_(sizes.size()), sizes_(sizes) {
            for (size_t i = 1; i < num_layers_; i++) {
                biases_.push_back(randn < mat > (sizes[i], 1));
                weights_.push_back(randn < mat > (sizes[i], sizes[i - 1]));
            }
            /*
            ifstream in_file("../data/wb");
            for (size_t i = 1; i < num_layers_; i++) {
                for (size_t row = 0; row < sizes[i]; row++) {
                    for (size_t col = 0; col < sizes[i - 1]; col++) {
                        double tmp;
                        in_file >> tmp;
                        weights_[i-1](row, col) = tmp;
                    }
                }
            }
            for (size_t i = 1; i < num_layers_; i++) {
                for (size_t row = 0; row < sizes[i]; row++) {
                    for (size_t col = 0; col < 1; col++) {
                        double tmp;
                        in_file >> tmp;
                        biases_[i-1](row, col) = tmp;
                    }
                }
            }
             */
        }

        void SGD(vector<pair<mat, mat>> &training_data,
                 size_t epochs, size_t mini_batch_size, double eta,
                 const vector<pair<mat, mat>> &test_data) {
            for (size_t i = 0; i < epochs; i++) {
                std::random_shuffle(training_data.begin(), training_data.end());
                for (size_t j = 0; j < training_data.size() / mini_batch_size; j++) {
                    update_mini_batch(training_data.begin() + j * mini_batch_size,
                                      min(training_data.end(), training_data.begin() + (j + 1) * mini_batch_size),
                                      eta);
                }
                cout << "Epoch {"
                     << i << "}: {"
                     << evaluate(test_data)
                     << "} / {"
                     << test_data.size()
                     << "}"
                     << endl;
            }
        }


    private:
        mat feedforward(mat a) const {
            for (size_t i = 0; i < weights_.size(); i++) {
                a = sigmoid(weights_[i] * a + biases_[i]);
            }
            return a;
        }

        void update_mini_batch(vector<pair<mat, mat>>::iterator begin,
                               const vector<pair<mat, mat>>::iterator end, double eta) {
            vector<mat> nabla_w;
            vector<mat> nabla_b;
            for (size_t i = 1; i < num_layers_; i++) {
                nabla_w.push_back(zeros(sizes_[i], sizes_[i - 1]));
                nabla_b.push_back(zeros(sizes_[i], 1));
            }
            for (auto it = begin; it < end; it++) {
                backprop(it->first, it->second, nabla_w, nabla_b);
            }
            for (size_t i = 0; i < weights_.size(); i++) {
                auto tmp = (eta / (end - begin)) * nabla_w[i];
                weights_[i] -= tmp;
            }
            for (size_t i = 0; i < biases_.size(); i++) {
                biases_[i] -= (eta / (end - begin)) * nabla_b[i];
            }
        }

        void backprop(const mat &x, const mat &y, vector<mat> &nabla_w, vector<mat> &nabla_b) {
            auto activation = x;
            vector<mat> activations;
            vector<mat> zs;
            activations.push_back(activation);
            for (size_t i = 0; i < weights_.size(); i++) {
                auto z = weights_[i] * activation + biases_[i];
                zs.push_back(z);
                activation = sigmoid(z);
                activations.push_back(activation);
            }
            mat delta = cost_derivative(activations.back(), y) % sigmoid_prime(zs.back());
            nabla_b.back() += delta;
            nabla_w.back() += delta * activations[activations.size() - 2].t();
            for (size_t i = 2; i < num_layers_; i++) {
                auto z = zs[zs.size() - i];
                auto sp = sigmoid_prime(z);
                delta = (weights_[weights_.size() - i + 1].t() * delta) % sp;
                nabla_b[nabla_b.size() - i] += delta;
                nabla_w[nabla_w.size() - i] += delta * activations[activations.size() - i - 1].t();
            }
        }

        size_t evaluate(const vector<pair<mat, mat>> &test_data) const {
            size_t result = 0;
            for (const auto &item : test_data) {
                auto max_index = feedforward(item.first).index_max();
                if ((int) item.second[max_index] == 1) result++;
            }
            return result;
        }

        mat cost_derivative(const mat &output_activations, const mat &y) const {
            return output_activations - y;
        }

        mat sigmoid(const mat &z) const {
            return 1.0 / (1.0 + exp(-z));
        }

        mat sigmoid_prime(const mat &z) const {
            return sigmoid(z) % (1 - sigmoid(z));
        }

        size_t num_layers_;
        const vector<int> sizes_;
        vector<mat> biases_;
        vector<mat> weights_;
    };
}
