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


// Probability distribution of Multivariate Gaussian
// Up to six independent variables
// Three spatial values
// Three velocity values
// Implement such that n samples can be used
template<typename T>
class MultivariateGaussian {

public:
    void evaluate(int nr_samples){};
    std::vector<T> get_samples(){return this->X;};
    std::vector<T> evaluate_and_get_samples(int nr_samples){this->evaluate(nr_samples); return this->X;};

    MultivariateGaussian(){};
    MultivariateGaussian(std::vector<T> mean, std::vector<T> covariance){
    this->mean = mean;
    this->covariance  = covariance;
  };

private:
    std::vector<T> X;
    std::vector<T> mean;
    std::vector<T> covariance;
};


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
    GaussianMixtureModel() {
    this->n_components = 10;
    this->n_features = 10;
    this->n_data = 10;
    std::cout << "constructor" << std::endl;};

    GaussianMixtureModel(size_t n_data) {
    this->n_data = n_data;
    std::cout << "n_data: " << n_data << std::endl;
    }

  private:
    std::vector<double> expectation(std::vector<double> weights);
    int maximization();

    std::vector<double> data;
    std::vector<double> mean;
    std::vector<double> covariance;
    std::vector<double> pi;
    size_t n_components;
    size_t n_features;
    size_t n_data;
};


/*
Expectation part of the EM algorithm
*/
std::vector<double> GaussianMixtureModel::expectation(std::vector<double> weights) {
  double numerator;

  MultivariateGaussian MvGaDs = MultivariateGaussian<double>(this->mean, this->covariance);
  for (int i = 0; i < this->n_components; i++) {

    MvGaDs.evaluate_and_get_samples(this->n_data);
    //std::normal_distribution n_d{mean, standard_deviation};
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
    //std::array<double, this->n_components*this->n_data> weights;
    //this->expectation(weights);
    this->maximization();

 };
}

int GaussianMixtureModel::evaluate() {return 0;}
int GaussianMixtureModel::warmstart_init() {return 0;}
int GaussianMixtureModel::random_init() {return 0;}


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
