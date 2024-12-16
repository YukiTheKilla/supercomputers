#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

using namespace std;

void striped_algorithm(vector<int>& A, vector<int>& B, vector<int>& C, int n, int mpi_rank, int mpi_size) {
    if (n % mpi_size != 0) {
        if (mpi_rank == 0) {
            cerr << "Error: Matrix size must be divisible by the number of MPI processes.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
        return;
    }

    int block_size = n / mpi_size;

    vector<int> local_A(block_size * n, 0);
    vector<int> local_B(n * n, 0);
    vector<int> local_C(block_size * n, 0);

    MPI_Scatter(A.data(), block_size * n, MPI_INT, local_A.data(), block_size * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }

    MPI_Gather(local_C.data(), block_size * n, MPI_INT, C.data(), block_size * n, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    vector<int> matrix_sizes = {128, 256, 512}; // Размеры матриц
    const int num_attempts = 10;               // Количество попыток

    for (int size : matrix_sizes) {
        int n = size; // Размер текущей матрицы
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, 10);

        vector<int> A(n * n);
        vector<int> B(n * n);
        vector<int> C(n * n, 0);

        if (mpi_rank == 0) {
            for (int i = 0; i < n * n; ++i) {
                A[i] = dis(gen);
                B[i] = dis(gen);
            }
        }

        double total_time = 0.0;

        for (int attempt = 0; attempt < num_attempts; ++attempt) {
            // Обнуление результирующей матрицы C перед каждым запуском
            fill(C.begin(), C.end(), 0);

            auto start = chrono::high_resolution_clock::now();
            striped_algorithm(A, B, C, n, mpi_rank, mpi_size);
            auto end = chrono::high_resolution_clock::now();

            if (mpi_rank == 0) {
                double duration = chrono::duration<double>(end - start).count();
                total_time += duration;
            }
        }

        if (mpi_rank == 0) {
            double average_time = total_time / num_attempts;
            cout << "Average time for size " << n << " over " << num_attempts 
                 << " attempts: " << average_time << " seconds\n";
        }
    }

    MPI_Finalize();
    return 0;
}