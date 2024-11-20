// #include <openPMD/openPMD.hpp>

// #include <mpi.h>

#include <cstddef>
#include <iostream>
#include <memory>

// Added for MGD
#include <random>

// Added for GMM
#include <cmath>
// #include <map>

#include <array>
#include <vector>

#include <chrono>

// using namespace openPMD;

template <typename T, std::size_t Row, std::size_t Col>
using mtx = std::array<std::array<T, Col>, Row>;

// Simple SPD Matrix Class
// Change to std::array (should be fine for now)
// To be wrapped over Eigen etc

// template<typename T, std::size_t m>
template <typename T, int m> // static std::size_t n>
class MTXSPD {

public:
  MTXSPD() {};
  // MTXSPD(std::array<std::array<T,n>,n> A, int m){this->m = m; this->A = A;
  // this->chol(); this->invert_L();}; // Some check that it is S in SPD;
  MTXSPD(std::vector<std::vector<T>> A) {
    this->A = A;
    chol();
    invert_L();
  }; // Some check that it is S in SPD;

  void apply_inv(std::vector<T> &vec) {
    this->apply_Linvtrans(vec);
    this->apply_Linv(vec);
  };
  void apply_Linv(std::vector<T> &vec);
  void apply_Linvtrans(std::vector<T> &vec);

  void apply_inv(std::array<std::vector<T>, m> &vec,
                 const std::vector<T> &mean) {
    this->apply_Linvtrans(vec);
    this->apply_Linv(vec, mean);
  };
  void apply_Linv(std::array<std::vector<T>, m> &vec,
                  const std::vector<T> &mean);
  void apply_Linvtrans(std::array<std::vector<T>, m> &vec);

  void print_A(int mat) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        if (mat == 1) {
          std::cout << this->A[i][j] << " ";
        } else if (mat == 2) {
          std::cout << this->L[i][j] << " ";
        } else if (mat == 3) {
          std::cout << this->Linv[i][j] << " ";
        }
      }
      std::cout << std::endl;
    };
  };

private:
  std::vector<std::vector<T>> A;
  std::vector<std::vector<T>> L;
  std::vector<std::vector<T>> Linv;
  void invert_L();
  void chol();
};

// should probably make mean an array<T,m>
template <class T, int m>
void MTXSPD<T, m>::apply_Linv(std::array<std::vector<T>, m> &vec,
                              const std::vector<T> &mean) {
  // #pragma omp parallel for
  for (int k = 0; k < vec[0].size(); k++) {
    for (int i = m - 1; i > -1; i--) {
      T res = 0.;
      for (int j = 0; j <= i; j++) {
        res += Linv[i][j] * vec[j][k];
      }
      vec[i][k] = res + mean[i];
    }
  }
};

// We apply L* on a vec in place
template <class T, int m>
void MTXSPD<T, m>::apply_Linvtrans(std::array<std::vector<T>, m> &vec) {
  // #pragma omp parallel for
  for (int k = 0; k < vec[0].size(); k++) {
    for (int i = 0; i < m; i++) {
      T res = 0.;
      for (int j = i; j < m; j++) {
        res += Linv[j][i] * vec[j][k];
      }
      vec[i][k] = res;
    }
  }
};

// We apply L on a vec in palce
template <class T, int m> void MTXSPD<T, m>::apply_Linv(std::vector<T> &vec) {

  for (int i = m - 1; i > -1; i--) {
    T res = 0.;
    for (int j = 0; j <= i; j++) {
      res += Linv[i][j] * vec[j];
    }
    vec[i] = res;
  }
};

// We apply L* on a vec in place
template <class T, int m>
void MTXSPD<T, m>::apply_Linvtrans(std::vector<T> &vec) {
  for (int i = 0; i < m; i++) {
    T res = 0.;
    for (int j = i; j < m; j++) {
      res += Linv[j][i] * vec[j];
    }
    vec[i] = res;
  }
};

// Cholesky–Banachiewicz algorithm
template <class T, int m> void MTXSPD<T, m>::chol() {
  std::vector<std::vector<T>> L(m, std::vector<T>(m));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j <= i; j++) {
      float sum = 0;
      // #pragma unroll
      for (int k = 0; k < j; k++)
        sum += L[i][k] * L[j][k];

      if (i == j)
        L[i][j] = sqrt(A[i][i] - sum);
      else
        L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum));
    }
  }
  this->L = L;
};

// Naive inverse
// Forward substition for
// each unit basis vector (b)
template <class T, int m> void MTXSPD<T, m>::invert_L() {
  std::vector<std::vector<T>> Linv(m, std::vector<T>(m));
  for (int b = 0; b < m; b++) {
    for (int i = 0; i < m; i++) {
      float sum = 0;
      for (int j = 0; j < i; j++) {
        sum += L[i][j] * Linv[j][b];
      }
      Linv[i][b] = ((T)(i == b) - sum) / L[i][i];
    }
  }
  this->Linv = Linv;
};

// Probability distribution of Multivariate Gaussian
// Up to six independent variables
// Three spatial values
// Three velocity values
// Implement such that n samples can be used
//
// template<typename T, std::size_t n_features>
template <typename T, std::size_t n_features> class MultivariateGaussian {
public:
  void evaluate(int nr_samples) {
    std::normal_distribution<T> unitary_normal_distribution{0, 1.};
    std::array<std::vector<T>, n_features> X;

    for (int i = 0; i < n_features; i++) {
      std::vector<T> x;
      for (int j = 0; j < nr_samples; j++) {
        x.push_back(unitary_normal_distribution(gen));
      }
      X[i] = x;
    }

    covariance.apply_Linvtrans(X);
    covariance.apply_Linv(X, mean);

    // for (int i = 0; i < n_features; i++) {
    //   for (int j = 0; j < nr_samples; j++) {

    //    std::cout << X[i][j] << " ";
    //  }
    //  std::cout << std::endl;
    //}
  };

  std::vector<T> get_samples() { return this->X; };
  std::vector<T> evaluate_and_get_samples(int nr_samples) {
    this->evaluate(nr_samples);
    return this->X;
  };

  MultivariateGaussian() {};
  MultivariateGaussian(std::mt19937 gen, std::vector<T> mean,
                       MTXSPD<T, n_features> covariance) {
    this->mean = mean;
    this->covariance = covariance;
  };

private:
  std::vector<T> X; // Random variable
  std::vector<T> mean;
  // std::array<T, n_features> mean;
  MTXSPD<T, n_features> covariance;
  std::mt19937 gen; // gen{rd()};
};

// Gaussian Mixture class
// EM algorithm
// template<typename T, std::size_t n_features, std::size_t n_components>
class GaussianMixtureModel {

public:
  // Public functions
  void train(int n_train);
  int evaluate();
  int warmstart_init();
  int random_init();
  int get_n_comp() { return this->n_components; };

  // Standard constructor
  GaussianMixtureModel() {
    this->n_components = 10;
    this->n_features = 10;
    this->n_data = 10;
    std::cout << "constructor" << std::endl;
  };

  GaussianMixtureModel(size_t n_data) {
    this->n_data = n_data;
    std::cout << "n_data: " << n_data << std::endl;
  }

private:
  std::vector<std::vector<double>>
  expectation(std::vector<std::vector<double>> weights);
  int maximization();

  std::vector<double> data;
  std::vector<double> mean;
  std::vector<MTXSPD<double, 3>> covariance;
  std::vector<double> pi;
  size_t n_components;
  size_t n_features;
  size_t n_data;
};

/*
Expectation part of the EM algorithm
*/
std::vector<std::vector<double>>
GaussianMixtureModel::expectation(std::vector<std::vector<double>> weights) {
  double numerator;

  // Should covariance be a list ov matrices or
  // should we work on a single pdf at the time?
  // MultivariateGaussian MvGaDs = MultivariateGaussian<double>(this->mean,
  // this->covariance); for (int i = 0; i < this->n_components; i++) {
  //  MvGaDs.evaluate_and_get_samples(this->n_data);

  // weights{i*n_data} =
  //   }
  // weights /= std::reduce(weights)

  double scale = 0;
  for (int i = 0; i < this->n_components; i++) {
    for (int j = 0; j < n_data; j++) {
      scale += 1; // weights{i*n_data};
    }
  }
  return weights;
}

int GaussianMixtureModel::maximization() { return 0; }

void GaussianMixtureModel::train(int n_train) {
  for (int i = 0; i < n_train; i++) {
    std::vector<std::vector<double>> weights;
    this->expectation(weights);
    this->maximization();
  };
}

int GaussianMixtureModel::evaluate() { return 0; }
int GaussianMixtureModel::warmstart_init() { return 0; }
int GaussianMixtureModel::random_init() { return 0; }

class AIC {};
class BIC {};
class AdaptiveGaussianMixtureModel {};
class DistributedGaussianMixtureModel {};

int main(int argc, char *argv[]) {

  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::cout << std::scientific;
  GaussianMixtureModel GMM = GaussianMixtureModel(10);
  GaussianMixtureModel GMM2 = GaussianMixtureModel();

  std::vector<std::vector<double>> A = {
      {4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
  MTXSPD<double, 3> TestMTX{A};
  std::cout << "-- A --" << std::endl;
  TestMTX.print_A(1);
  std::cout << "-- L --" << std::endl;
  TestMTX.print_A(2);
  std::cout << "-- Linv --" << std::endl;
  TestMTX.print_A(3);
  std::vector<double> x = {1, 1, 1};
  TestMTX.apply_Linv(x);
  std::cout << "____" << std::endl;
  std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
  x = {1, 1, 1};
  TestMTX.apply_Linvtrans(x);
  std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
  x = {1, 1, 1};
  TestMTX.apply_inv(x);
  std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
  std::cout << "____" << std::endl;

  MTXSPD<double, 3> cov{A};
  std::vector<double> mean{1., 2., 3.};

  auto t1 = std::chrono::high_resolution_clock::now();
  MultivariateGaussian<double, 3> MG{gen, mean, cov};
  int a = 1e7;
  MG.evaluate(a);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Multivariate Gaussian Evaluate took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " milliseconds\n";

  // MG.evaluate(a);
}
