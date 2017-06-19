//
// network.hpp
//
// A module to implement the stochastic gradient descent learning
// algorithm for a feedforward neural network.  Gradients are calculated
// using backpropagation.  Note that I have focused on making the code
// simple, easily readable, and easily modifiable.  It is not optimized,
// and omits many desirable features.


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
            //   The vector ``sizes`` contains the number of neurons in the
            // respective layers of the network.  For example, if the vector
            // was {2, 3, 1} then it would be a three-layer network, with the
            // first layer containing 2 neurons, the second layer 3 neurons,
            // and the third layer 1 neuron.  The biases and weights for the
            // network are initialized randomly, using a Gaussian
            // distribution with mean 0, and variance 1.  Note that the first
            // layer is assumed to be an input layer, and by convention we
            // won't set any biases for those neurons, since biases are only
            // ever used in computing the outputs from later layers.

            for (size_t i = 1; i < num_layers_; i++) {
                biases_.push_back(randn < mat > (sizes[i], 1));
                weights_.push_back(randn < mat > (sizes[i], sizes[i - 1]));
            }
        }

        void SGD(vector<pair<mat, mat>> &training_data,
                 size_t epochs, size_t mini_batch_size, double eta,
                 const vector<pair<mat, mat>> &test_data) {
            //   Train the neural network using mini-batch stochastic
            // gradient descent.  The ``training_data`` is a vector of pair
            // (x, y) representing the training inputs and the desired
            // outputs.  The other non-optional parameters are self-explanatory.
            // network will be evaluated against the test data after each
            // epoch, and partial progress printed out.  This is useful for
            // tracking progress, but slows things down substantially.

            for (size_t i = 0; i < epochs; i++) {
                std::random_shuffle(training_data.begin(), training_data.end());
                for (size_t j = 0; j < training_data.size() / mini_batch_size; j++) {
                    update_mini_batch(training_data.begin() + j * mini_batch_size,
                                      min(training_data.end(), training_data.begin() + (j + 1) * mini_batch_size),
                                      eta);
                }
                cout << "Epoch "
                     << i << ": "
                     << evaluate(test_data)
                     << " / "
                     << test_data.size()
                     << ""
                     << endl;
            }
        }


    private:
        mat feedforward(mat a) const {
            // Return the output of the network if ``a`` is input.
            for (size_t i = 0; i < weights_.size(); i++) {
                a = sigmoid(weights_[i] * a + biases_[i]);
            }
            return a;
        }

        void update_mini_batch(vector<pair<mat, mat>>::iterator begin,
                               const vector<pair<mat, mat>>::iterator end, double eta) {
            //   Update the network's weights and biases by applying
            // gradient descent using backpropagation to a single mini batch.
            // The ``mini_batch`` is a pair of iterators (begin, end) and ``eta``
            // is the learning rate."""
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
            //   Return  ``(nabla_b, nabla_w)`` representing the
            // gradient for the cost function C_x.  ``nabla_b`` and
            // ``nabla_w`` are layer-by-layer vector of numpy arrays, similar
            // to ``self.biases`` and ``self.weights``.
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
            //   Return the number of test inputs for which the neural
            // network outputs the correct result. Note that the neural
            // network's output is assumed to be the index of whichever
            // neuron in the final layer has the highest activation.
            size_t result = 0;
            for (const auto &item : test_data) {
                auto max_index = feedforward(item.first).index_max();
                if ((int) item.second[max_index] == 1) result++;
            }
            return result;
        }

        mat cost_derivative(const mat &output_activations, const mat &y) const {
            //   Return the partial derivatives \partial C_x /
            //partial a for the output activations.
            return output_activations - y;
        }

        mat sigmoid(const mat &z) const {
            // The sigmoid function.
            return 1.0 / (1.0 + exp(-z));
        }

        mat sigmoid_prime(const mat &z) const {
            // Derivative of the sigmoid function.
            return sigmoid(z) % (1 - sigmoid(z));
        }

        size_t num_layers_;
        const vector<int> sizes_;
        vector<mat> biases_;
        vector<mat> weights_;
    };
}
