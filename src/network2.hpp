//
// network.hpp
//
//   An improved version of network.py, implementing the stochastic
// gradient descent learning algorithm for a feedforward neural network.
// Improvements include the addition of the cross-entropy cost function,
// regularization, and better initialization of network weights.  Note
// that I have focused on making the code simple, easily readable, and
// easily modifiable.  It is not optimized, and omits many desirable
// features.
//
#ifndef DEEPLEARNINGCPP_NETWORK2_HPP
#define DEEPLEARNINGCPP_NETWORK2_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <armadillo>

namespace deep_learning_cpp {
    namespace network2 {

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

        mat sigmoid(const mat &z) {
            // The sigmoid function.
            return 1.0 / (1.0 + exp(-z));
        }

        mat sigmoid_prime(const mat &z) {
            // Derivative of the sigmoid function.
            return sigmoid(z) % (1 - sigmoid(z));
        }

        class QuadraticCost {
        public:
            static double fn(const mat &a, const mat &y) {
                // Return the cost associated with an output ``a`` and desired output
                return 0.5 * std::pow(arma::norm(a - y, 2), 2);
            }

            static mat delta(const mat &z, const mat &a, const mat &y) {
                // Return the error delta from the output layer."""
                return (a - y) % sigmoid_prime(z);
            }

        };

        class CrossEntropyCost {
        public:
            static double fn(const mat &a, const mat &y) {
                //   Return the cost associated with an output ``a`` and desired output
                // ``y``.  Note that np.nan_to_num is used to ensure numerical
                // stability.  In particular, if both ``a`` and ``y`` have a 1.0
                // in the same slot, then the expression (1-y)*np.log(1-a)
                // returns nan.  The np.nan_to_num ensures that that is converted
                // to the correct value (0.0).
                mat tmp = -y % arma::log(a) - (1 - y) % arma::log(1 - a);
                tmp.elem(arma::find_nonfinite(tmp)).zeros();
                return arma::accu(tmp);
            }

            static mat delta(const mat &z, const mat &a, const mat &y) {
                //   Return the error delta from the output layer.  Note that the
                // parameter ``z`` is not used by the method.  It is included in
                // the method's parameters in order to make the interface
                // consistent with the delta method for other cost classes.
                return a - y;
            }
        };

        enum cost_enum {
            CrossEntropyCost,
            QuadraticCost,
        };

        class Network2 {
        public:
            Network2() = default;

            Network2(const vector<int> &sizes, cost_enum cost = CrossEntropyCost) :
                    num_layers_(sizes.size()),
                    sizes_(sizes),
                    cost_(cost) {
                //   The vector ``sizes`` contains the number of neurons in the respective
                // layers of the network.  For example, if the list was [2, 3, 1]
                // then it would be a three-layer network, with the first layer
                // containing 2 neurons, the second layer 3 neurons, and the
                // third layer 1 neuron.  The biases and weights for the network
                // are initialized randomly, using
                // ``self.default_weight_initializer`` (see docstring for that
                // method).

                default_weight_initializer();
            }

            std::tuple<vector<double>, vector<size_t>, vector<double>, vector<size_t>>
            SGD(vector<pair<mat, mat>> &training_data,
                size_t epochs, size_t mini_batch_size, double eta,
                double lmbda,
                const vector<pair<mat, mat>> &evaluation_data,
                bool monitor_evaluation_cost = false,
                bool monitor_evaluation_accuracy = false,
                bool monitor_training_cost = false,
                bool monitor_training_accuracy = false) {
                //   Train the neural network using mini-batch stochastic gradient
                // descent.  The ``training_data`` is a vector of pairs ``(x, y)``
                // representing the training inputs and the desired outputs.  The
                // other non-optional parameters are self-explanatory, as is the
                // regularization parameter ``lmbda``.  The method also accepts
                // evaluation_data``, usually either the validation or test
                // data.  We can monitor the cost and accuracy on either the
                // evaluation data or the training data, by setting the
                // appropriate flags.  The method returns a tuple containing four
                // vectors: the (per-epoch) costs on the evaluation data, the
                // accuracies on the evaluation data, the costs on the training
                // data, and the accuracies on the training data.  All values are
                // evaluated at the end of each training epoch.  So, for example,
                // if we train for 30 epochs, then the first element of the tuple
                // will be a 30-element list containing the cost on the
                // evaluation data at the end of each epoch. Note that the lists
                // are empty if the corresponding flag is not set.
                vector<double> evaluation_cost, training_cost;
                vector<size_t> evaluation_accuracy, training_accuracy;
                for (size_t i = 0; i < epochs; i++) {
                    std::random_shuffle(training_data.begin(), training_data.end());
                    for (size_t j = 0; j < training_data.size() / mini_batch_size; j++) {
                        update_mini_batch(training_data.begin() + j * mini_batch_size,
                                          min(training_data.end(), training_data.begin() + (j + 1) * mini_batch_size),
                                          lmbda,
                                          eta,
                                          training_data.size());
                    }
                    cout << "Epoch "
                         << i << " training complete"
                         << endl;
                    if (monitor_training_cost) {
                        auto cost = total_cost(training_data, lmbda);
                        training_cost.push_back(cost);
                        cout << "Cost on training data: " << cost << endl;
                    }
                    if (monitor_training_accuracy) {
                        auto acc = accuracy(training_data);
                        training_accuracy.push_back(acc);
                        cout << "Accuracy on training data: "
                             << acc
                             << " / "
                             << training_data.size()
                             << endl;
                    }
                    if (monitor_evaluation_cost) {
                        auto cost = total_cost(evaluation_data, lmbda);
                        evaluation_cost.push_back(cost);
                        cout << "Cost on evaluation data: " << cost << endl;
                    }
                    if (monitor_evaluation_accuracy) {
                        auto acc = accuracy(evaluation_data);
                        evaluation_accuracy.push_back(acc);
                        cout << "Accuracy on evaluation data: "
                             << acc
                             << " / "
                             << evaluation_data.size()
                             << endl;
                    }
                    cout << endl;
                }
                return std::make_tuple(evaluation_cost, evaluation_accuracy, training_cost, training_accuracy);
            }

            void save(const std::string &filename) {
                // Save the neural network to the file ``filename``.
                std::ofstream out_file(filename, std::ios::binary);
                out_file.write((char *) (this), sizeof(*this));
                out_file.close();
            }


        private:
            void default_weight_initializer() {
                //   Initialize each weight using a Gaussian distribution with mean 0
                // and standard deviation 1 over the square root of the number of
                // weights connecting to the same neuron.  Initialize the biases
                // using a Gaussian distribution with mean 0 and standard
                // deviation 1.

                // Note that the first layer is assumed to be an input layer, and
                // by convention we won't set any biases for those neurons, since
                // biases are only ever used in computing the outputs from later
                // layers.
                for (size_t i = 1; i < num_layers_; i++) {
                    biases_.push_back(randn < mat > (sizes_[i], 1));
                    weights_.push_back(randn < mat > (sizes_[i], sizes_[i - 1]) / std::sqrt(sizes_[i - 1]));
                }
            }

            void large_weight_initializer() {
                //   Initialize the weights using a Gaussian distribution with mean 0
                // and standard deviation 1.  Initialize the biases using a
                // Gaussian distribution with mean 0 and standard deviation 1.

                // Note that the first layer is assumed to be an input layer, and
                // by convention we won't set any biases for those neurons, since
                // biases are only ever used in computing the outputs from later
                // layers.

                //   This weight and bias initializer uses the same approach as in
                // Chapter 1, and is included for purposes of comparison.  It
                // will usually be better to use the default weight initializer
                // instead.
                for (size_t i = 1; i < num_layers_; i++) {
                    biases_.push_back(randn < mat > (sizes_[i], 1));
                    weights_.push_back(randn < mat > (sizes_[i], sizes_[i - 1]));
                }
            }

            mat feedforward(mat a) const {
                // Return the output of the network if ``a`` is input.
                for (size_t i = 0; i < weights_.size(); i++) {
                    a = sigmoid(weights_[i] * a + biases_[i]);
                }
                return a;
            }

            void update_mini_batch(vector<pair<mat, mat>>::iterator begin,
                                   const vector<pair<mat, mat>>::iterator end,
                                   double lmbda,
                                   double eta,
                                   size_t n) {
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
                    weights_[i] *= (1 - eta * (lmbda / n));
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

                // feedforward
                auto activation = x;
                vector<mat> activations; // vector to store all the activations, layer by layer
                vector<mat> zs; // vector to store all the z vectors, layer by layer
                activations.push_back(activation);
                for (size_t i = 0; i < weights_.size(); i++) {
                    auto z = weights_[i] * activation + biases_[i];
                    zs.push_back(z);
                    activation = sigmoid(z);
                    activations.push_back(activation);
                }
                // backward pass
                mat delta;
                if (cost_ == CrossEntropyCost) {
                    delta = CrossEntropyCost::delta(zs.back(), activations.back(), y);
                } else if (cost_ == QuadraticCost) {
                    delta = QuadraticCost::delta(zs.back(), activations.back(), y);

                }
                nabla_b.back() += delta;
                nabla_w.back() += delta * activations[activations.size() - 2].t();
                // Note that the variable l in the loop below is used a little
                // differently to the notation in Chapter 2 of the book.  Here,
                // l = 1 means the last layer of neurons, l = 2 is the
                // second-last layer, and so on.
                for (size_t l = 2; l < num_layers_; l++) {
                    auto z = zs[zs.size() - l];
                    auto sp = sigmoid_prime(z);
                    delta = (weights_[weights_.size() - l + 1].t() * delta) % sp;
                    nabla_b[nabla_b.size() - l] += delta;
                    nabla_w[nabla_w.size() - l] += delta * activations[activations.size() - l - 1].t();
                }
            }

            size_t accuracy(const vector<pair<mat, mat>> &data) const {
                //   Return the number of inputs in ``data`` for which the neural
                // network outputs the correct result. The neural network's
                // output is assumed to be the index of whichever neuron in the
                // final layer has the highest activation.

                size_t result = 0;
                for (const auto &item : data) {
                    auto max_index = feedforward(item.first).index_max();
                    if ((int) item.second[max_index] == 1) result++;
                }
                return result;
            }

            double total_cost(const vector<pair<mat, mat>> &data, double lmbda) const {
                //   Return the total cost for the data set ``data``.
                double cost = 0.0;
                for (const auto &item : data) {
                    auto a = feedforward(item.first);
                    if (cost_ == CrossEntropyCost) {
                        cost += (CrossEntropyCost::fn(a, item.second) / data.size());
                    } else if (cost_ == QuadraticCost) {
                        cost += (QuadraticCost::fn(a, item.second) / data.size());
                    }
                }
                double sum = 0;
                for (const auto &w : weights_) {
                    sum += std::pow(arma::norm(w, 2), 2);
                }
                cost += 0.5 * (lmbda / data.size()) * sum;
                return cost;
            }

            size_t num_layers_;
            const vector<int> sizes_;
            vector<mat> biases_;
            vector<mat> weights_;
            cost_enum cost_;
        };

        Network2 load(const std::string &filename) {
            //   Load a neural network from the file ``filename``.  Returns an
            // instance of Network.

            std::ifstream in_file(filename, std::ios::binary);
            Network2 network;
            while (in_file.read((char *) &network, sizeof(network)));
            in_file.close();
            return network;
        }
    }
}


#endif //DEEPLEARNINGCPP_NETWORK2_HPP
