#include <openPMD/openPMD.hpp>

#include <mpi.h>

#include <cstddef>
//#include <iostream>
#include <memory>

// Added for MGD
#include <random>

// Added for GMM
#include <cmath>
//#include <map>

//using std::cout;
using namespace openPMD;


// Simple SPD Matrix Class
// Change to std::array (should be fine for now)
// To be wrapped over Eigen etc
template<typename T>
class MTXSPD {

public:
  MTXSPD(){};
  MTXSPD(std::vector<std::vector<T>> A, int m){this->m = m; this->A = A; this->chol(); this->invert_L();}; // Some check that it is S in SPD;
  MTXSPD<T> apply_inv(MTXSPD<T> matrix);
  std::vector<T> apply_inv(std::vector<T> vec){return this->apply_Linvtrans(this->apply_Linv(vec));};

  std::vector<T> apply_Linv(std::vector<T> vec);
  std::vector<T> apply_Linvtrans(std::vector<T> vec);


void print_A(int mat){ for (int i = 0; i < m; i++) {
                    for (int j = 0; j < m; j++) {
                        if (mat == 1) {
                        std::cout << this->A[i][j] << " ";
                        } else if (mat == 2) {
                        std::cout << this->L[i][j] << " ";
                        } else if (mat ==  3) {
                        std::cout << this->Linv[i][j] << " ";}
}
std::cout << std::endl; };
};


private:
  int m;
  std::vector<std::vector<T>> A;
  std::vector<std::vector<T>> L;
  std::vector<std::vector<T>> Linv;
  void invert_L();
  void chol();

};


// We apply L on a vec in palce
template<class T>
std::vector<T> MTXSPD<T>::apply_Linv(std::vector<T> vec){

  for (int i = m-1;  i > -1; i--) {
    T res = 0.;
    for (int j = 0;  j <= i; j++) {
      res += Linv[i][j]*vec[j];
    }
    vec[i] = res;
  }
return vec;
};

// We apply L* on a vec in place
template<class T>
std::vector<T> MTXSPD<T>::apply_Linvtrans(std::vector<T> vec){
  for (int i = 0; i < m; i++) {
    T res = 0.;
    for (int j = i;  j < m; j++) {
      res += Linv[j][i]*vec[j];
    }
    vec[i] = res;
  }
return vec;

};

//Cholesky–Banachiewicz algorithm
template<class T>
void MTXSPD<T>::chol(){
    std::vector<std::vector<T>> L(m,std::vector<T>(m));
    for (int i = 0; i < m; i++) {
    for (int j = 0; j <= i; j++) {
        float sum = 0;
//#pragma unroll
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
template<class T>
void MTXSPD<T>::invert_L(){
    std::vector<std::vector<T>> Linv(this->m,std::vector<T>(this->m));
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
template<typename T>
class MultivariateGaussian {

public:
    void evaluate(int nr_samples){
    std::normal_distribution<> unitary_normal_distribution {0, 1.};
		// X unitary_normal_distribution.get_samples(nr_samples)
		//
};

    std::vector<T> get_samples(){return this->X;};
    std::vector<T> evaluate_and_get_samples(int nr_samples){this->evaluate(nr_samples); return this->X;};
		
/*
		for (int i = 0; i <  this->n_features ; i++) {
			for (int j = 0; j < nr_samples ; j++) {
				
			}
			
		}
*/
    MultivariateGaussian(){};
    MultivariateGaussian(std::vector<T> mean, std::vector<MTXSPD<T>> covariance){
    this->mean = mean;
    this->covariance = covariance;
  };

private:
    std::vector<T> X; // Random variable
    std::vector<T> mean;
    std::vector<MTXSPD<T>> covariance;
    int n_features; // 2-6 variables


};


//MultivariateGaussian::evaluate_and_get_samples()


// Gaussian Mixture class
//
//
//
class GaussianMixtureModel {

  public:
    // Public functions
    void train(int n_train);
    int evaluate();
    int warmstart_init();
    int random_init();
    int get_n_comp(){return this->n_components;};

    // Standard constructor
    GaussianMixtureModel() {this->n_components = 10;this->n_features = 10;this->n_data = 10;std::cout << "constructor" << std::endl;};
    GaussianMixtureModel(size_t n_data) {this->n_data = n_data;std::cout << "n_data: " << n_data << std::endl;}

  private:
    std::vector<std::vector<double>> expectation(std::vector<std::vector<double>> weights);
    int maximization();

    std::vector<double> data;
    std::vector<double> mean;
    std::vector<MTXSPD<double>> covariance;
    std::vector<double> pi;
    size_t n_components;
    size_t n_features;
    size_t n_data;
};


/*
Expectation part of the EM algorithm
*/
std::vector<std::vector<double>> GaussianMixtureModel::expectation(std::vector<std::vector<double>> weights) {
  double numerator;

  // Should covariance be a list ov matrices or
  // should we work on a single pdf at the time?
  MultivariateGaussian MvGaDs = MultivariateGaussian<double>(this->mean, this->covariance);
  for (int i = 0; i < this->n_components; i++) {
    MvGaDs.evaluate_and_get_samples(this->n_data);

//weights{i*n_data} =
  }
  //weights /= std::reduce(weights)

  double scale = 0;
  for (int i = 0;  i < this->n_components; i++) {
    for (int j = 0;  j < n_data; j++) {
      scale += 1; //weights{i*n_data};
    }
  }
  return weights;
}

int GaussianMixtureModel::maximization() {return 0;}

void GaussianMixtureModel::train(int n_train) {
  for (int i = 0; i < n_train; i++) {
    //std::vector<double> weights;
    std::vector<std::vector<double>> weights;
    this->expectation(weights);
    this->maximization();

 };
}

int GaussianMixtureModel::evaluate() {return 0;}
int GaussianMixtureModel::warmstart_init() {return 0;}
int GaussianMixtureModel::random_init() {return 0;}


class AIC{};
class BIC{};
class AdaptiveGaussianMixtureModel{};
class DistributedGaussianMixtureModel{};



int main(int argc, char *argv[])
{

  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::cout << std::scientific;
  GaussianMixtureModel GMM = GaussianMixtureModel(10);
  GaussianMixtureModel GMM2 = GaussianMixtureModel();
  std::cout << "hihi " << GMM2.get_n_comp() << std::endl;
  std::cout << "hihi" << std::endl;

  MultivariateGaussian<double> MG = MultivariateGaussian<double>();
  std::vector<std::vector<double>> Mat = {{0, 2, 3}, {4, 5, 6}};
  std::cout << "Mat[0][3] = " << Mat[0][2]  << std::endl;

  std::vector<MultivariateGaussian<double>> MultivariateGaussianVec;

  std::vector<std::vector<double>> A = {{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
  MTXSPD<double> TestMTX(A, 3);

	std::cout << "-- A --" << std::endl;
  TestMTX.print_A(1);
	std::cout << "-- L --" << std::endl;
  TestMTX.print_A(2);
	std::cout << "-- Linv --" << std::endl;
	TestMTX.print_A(3);
  std::vector<double> x = {1, 1, 1};
  x = TestMTX.apply_Linv(x);
  std::cout << "____" << std::endl;
  std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
  x = {1, 1, 1};
  x = TestMTX.apply_Linvtrans(x);
  std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
  x = {1, 1, 1};
  x = TestMTX.apply_inv(x);
  std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;








//    MPI_Init(&argc, &argv);
//
//    int mpi_size;
//    int mpi_rank;
//
//    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
//    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
//
//    Series series = Series(
//        "../samples/git-sample/data%T.h5", Access::READ_ONLY, MPI_COMM_WORLD);
//    if (0 == mpi_rank)
//        cout << "Read a series in parallel with " << mpi_size << " MPI ranks\n";
//
//    MeshRecordComponent E_x = series.iterations[100].meshes["E"]["x"];
//
//    Offset chunk_offset = {static_cast<long unsigned int>(mpi_rank) + 1, 1, 1};
//    Extent chunk_extent = {2, 2, 1};
//
//    // If you know the datatype, use `loadChunk<double>(...)` instead.
//    auto chunk_data = E_x.loadChunkVariant(chunk_offset, chunk_extent);
//
//    if (0 == mpi_rank)
//        cout << "Queued the loading of a single chunk per MPI rank from "
//                "disk, "
//                "ready to execute\n";
//
//    // The iteration can be closed in order to help free up resources.
//    // The iteration's content will be flushed automatically.
//    // An iteration once closed cannot (yet) be reopened.
//    series.iterations[100].close();
//
//    if (0 == mpi_rank)
//        cout << "Chunks have been read from disk\n";
//
//    for (int i = 0; i < mpi_size; ++i)
//    {
//        if (i == mpi_rank)
//        {
//            cout << "Rank " << mpi_rank << " - Read chunk contains:\n";
//            for (size_t row = 0; row < chunk_extent[0]; ++row)
//            {
//                for (size_t col = 0; col < chunk_extent[1]; ++col)
//                {
//                    cout << "\t" << '(' << row + chunk_offset[0] << '|'
//                         << col + chunk_offset[1] << '|' << 1 << ")\t";
//                    /*
//                     * For hot loops, the std::visit(...) call should be moved
//                     * further up.
//                     */
//                    std::visit(
//                        [row, col, &chunk_extent](auto &shared_ptr) {
//                            cout << shared_ptr
//                                        .get()[row * chunk_extent[1] + col];
//                        },
//                        chunk_data);
//                }
//                cout << std::endl;
//            }
//        }
//
//        // this barrier is not necessary but structures the example output
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
//    // The files in 'series' are still open until the series is closed, at which
//    // time it cleanly flushes and closes all open file handles.
//    // One can close the object explicitly to trigger this.
//    // Alternatively, this will automatically happen once the garbage collector
//    // claims (every copy of) the series object.
//    // In any case, this must happen before MPI_Finalize() is called
//    series.close();
//
//    // openPMD::Series MUST be destructed or closed at this point
//    MPI_Finalize();
//
//    return 0;
}
