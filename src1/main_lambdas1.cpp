#include <iostream>
#include <vector>
#include <algorithm> 
#include <fstream>
#include <cmath>
#include "posit.h" // include non-tabulated posits
#include <unistd.h>
#include <cstdint>

using real = posit::Posit<int8_t, 8, 0, uint32_t, posit::PositSpec::WithInfs>;
//using real = posit::Posit<int16_t, 16, 2, uint32_t, posit::PositSpec::WithInfs>;
//using real = posit::Posit<int32_t, 32, 4, uint64_t, posit::PositSpec::WithInfs>;

// real numbers error tolerance
const double epsilon = 1e-5;
// parameter of the optimization problem
double C = 1;

template <typename T>
struct svm_parameters {
    std::vector<T> w;
    T b;
};

// Function used to read variables mu and eta from a given file and compute their difference to obtain lambdas
template <typename T>
void get_lambdas_from_mu_eta(std::string &filename, std::vector<T> &ret) {
    std::string line;
    std::ifstream file(filename);
    if (file.is_open()) {
        while (std::getline(file, line)) {
            double tmp;
            std::stringstream ss(line);
            // Read mu
            std::getline(ss, line, ',');
            std::stringstream ss1(line);
            ss1 >> tmp;
            T mu(tmp);
            // Read eta
            std::getline(ss, line, ',');
            std::stringstream ss2(line);
            ss2 >> tmp;
            T eta(tmp);
            T diff = mu - eta; // lambda
            ret.push_back(diff);
        }
        file.close();
    } else {
        std::cout << "Unable to open file" << std::endl;
    }
}

template <typename T>
T dot_product(const std::vector<T> &x1, const std::vector<T> &x2) {
    T res = T(0);
    if (x1.size() != x2.size()) {
        std::cerr << "Inner product requires the same number of elements." << std::endl;
        exit(-1);
    }
    for (int i = 0; i < x1.size(); i++) {
        res += x1[i] * x2[i];
    }
    return res;
}

// Function to compute square root of a posit number
template <typename T>
T posit_sqrt(T x) {
    double tmp = static_cast<double>(x);
    return T(std::sqrt(tmp));
}

// Function to compute exponential of a posit number
template <typename T>
T posit_exp(T x) {
    double tmp = static_cast<double>(x);
    return T(std::exp(tmp));
}

// Function used to implement a gaussian kernel
template <typename T>
T kernel(std::vector<T> x, std::vector<T> xi, float gamma) {
    std::vector<T> diff(x.size());
    for (int i = 0; i < x.size(); ++i) {
        diff[i] = (x[i] - xi[i]);
    }
    T norm = posit_sqrt(dot_product(diff, diff));
    return posit_exp(-T(gamma) * norm * norm);
}

template <typename T>
void read_dataset(std::string filename, std::vector<std::vector<T>> &ret) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error opening file" << std::endl;
        return;
    }
    std::string line;
    while (getline(file, line)) {
        std::vector<T> row;
        std::stringstream ss(line);
        std::string cell;
        double tmp;
        while (getline(ss, cell, ',')) {
            try {
                std::stringstream ss1(cell);
                ss1 >> tmp;
                row.push_back(T(tmp));
            } catch (const std::exception &e) {
                std::cout << "Error: " << e.what() << std::endl;
                return;
            }
        }
        ret.push_back(row);
    }
    file.close();
}

// Function used to perform inference with a single sample and non-linear kernel using only real (posit type)
template <typename T>
T predict_non_linear_point(std::vector<std::vector<T>> x, std::vector<T> y, std::vector<T> lambdas, std::vector<T> x_test) {
    int index = -1;
    T epsilonDouble = T(1e-2);
    T b = -T(1);
    for (int i = 0; i < x.size(); ++i) {
        if (lambdas[i] >= epsilonDouble && lambdas[i] <= T(C) - epsilonDouble) {
            b = T(1) / y[i];
            index = i;
            break;
        }
    }
    for (int i = 0; i < x.size(); ++i) {
        b = b - lambdas[i] * y[i] * kernel(x[index], x[i], 0.01);
    }
    T result = T(0);
    for (int i = 0; i < x.size(); ++i) {
        result = result + lambdas[i] * y[i] * kernel(x[i], x_test, 0.01);
    }
    result = result + b;
    T threshold = T(0);
    if (result >= threshold) {
        return T(+1);
    } else {
        return T(-1);
    }
}

// Function used to perform inference with the overall dataset in case of non-linear kernels
template <typename T>
std::vector<T> predict_non_linear_dataset(std::vector<std::vector<T>> x_test, std::vector<std::vector<T>> x, std::vector<T> y, std::vector<T> lambdas) {
    std::vector<T> classes;
    for (int i = 0; i < x_test.size(); i++)
        classes.push_back(predict_non_linear_point(x, y, lambdas, x_test[i]));
    return classes;
}

// Function used to compute accuracy
template <typename T>
float model_evaluate(std::vector<T> p, std::vector<T> y) {
    float res = 0;
    if (p.size() != y.size()) {
        std::cerr << "Accuracy computation requires the same number of elements." << std::endl;
        exit(-1);
    }
    for (int i = 0; i < p.size(); i++) {
        res += (p[i] == y[i]);
    }
    return res / p.size();
}

int main() {
    std::string filename = "../matlab/mueta_exp4.csv";
    // should be a file containing mus and etas or lambdas, and the corresponding function should be properly chosen below
    std::string train_data_filename = "../datasets/breast_cancer_train_normalized.csv";
    std::string test_data_filename = "../datasets/breast_cancer_test_normalized.csv";

    std::vector<real> lambdas;

    // Compute the support vectors from the mu and eta stored in the file
    get_lambdas_from_mu_eta(filename, lambdas);

    // reads the training and test sets
    std::vector<std::vector<double>> training_set, test_set;
    read_dataset(train_data_filename, training_set);
    read_dataset(test_data_filename, test_set);

    if (training_set.empty() || test_set.empty()) {
        std::cout << "Error reading the dataset" << std::endl;
        return -1;
    }

    // training set
    std::vector<std::vector<real>> X_train;
    std::vector<real> y_train;
    real elem;
    for (int i = 0; i < training_set.size(); i++) {
        std::vector<real> row;
        for (int j = 0; j < training_set[i].size() - 1; j++) {  // exclude the label
            elem = training_set[i][j];
            row.push_back(elem);
        }
        X_train.push_back(row);
    }

    for (int i = 0; i < training_set.size(); i++) {
        elem = training_set[i][training_set[i].size() - 1]; // save in y just the label
        y_train.push_back(elem);
    }

    // Test set
    std::vector<std::vector<real>> X_test;
    std::vector<real> y_test;

    for (int i = 0; i < test_set.size(); i++) {
        std::vector<real> row;
        for (int j = 0; j < test_set[i].size() - 1; j++) { // exclude the label
            elem = test_set[i][j];
            row.push_back(elem);
        }
        X_test.push_back(row);
    }

    for (int i = 0; i < test_set.size(); i++) {
        elem = test_set[i][test_set[i].size() - 1]; // save in y just the label
        y_test.push_back(elem);
    }

    std::vector<real> p = predict_non_linear_dataset(X_test, X_train, y_train, lambdas);
    float acc = model_evaluate(p, y_test);
    std::cout << "Accuracy: " << acc << std::endl;

    return 0;
}