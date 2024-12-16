#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <mpi.h>

using namespace std;

void fox_algorithm(vector<int>& A, vector<int>& B, vector<int>& C, int n, int mpi_rank, int mpi_size) {
    int sqrt_size = static_cast<int>(sqrt(mpi_size));
    if (sqrt_size * sqrt_size != mpi_size) {
        if (mpi_rank == 0) {
            cerr << "Error: The number of MPI processes must be a perfect square.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
        return;
    }

    int block_size = n / sqrt_size;

    vector<int> local_A(block_size * block_size, 0);
    vector<int> local_B(block_size * block_size, 0);
    vector<int> local_C(block_size * block_size, 0);
    vector<int> temp_A(block_size * block_size, 0);

    MPI_Comm grid_comm, row_comm, col_comm;
    int dims[2] = {sqrt_size, sqrt_size};
    int periods[2] = {0, 0};
    int coords[2];

    // Создание декартовой топологии
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    if (grid_comm == MPI_COMM_NULL) {
        cerr << "Error: Failed to create Cartesian communicator.\n";
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
        return;
    }

    MPI_Cart_coords(grid_comm, mpi_rank, 2, coords);

    int row_rank, col_rank;
    MPI_Comm_split(grid_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(grid_comm, coords[1], coords[0], &col_comm);

    // Распределение блоков
    MPI_Scatter(A.data(), block_size * block_size, MPI_INT, local_A.data(), block_size * block_size, MPI_INT, 0, grid_comm);
    MPI_Scatter(B.data(), block_size * block_size, MPI_INT, local_B.data(), block_size * block_size, MPI_INT, 0, grid_comm);

    for (int step = 0; step < sqrt_size; ++step) {
        int broadcast_root = (coords[0] + step) % sqrt_size;

        if (coords[1] == broadcast_root) {
            temp_A = local_A;
        }

        // Широковещательная передача блока A по строке
        MPI_Bcast(temp_A.data(), block_size * block_size, MPI_INT, broadcast_root, row_comm);

        // Умножение текущих блоков
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    local_C[i * block_size + j] += temp_A[i * block_size + k] * local_B[k * block_size + j];
                }
            }
        }

        // Сдвиг матрицы B вверх
        MPI_Sendrecv_replace(local_B.data(), block_size * block_size, MPI_INT,
                             (coords[0] + 1) % sqrt_size, 0,
                             (coords[0] - 1 + sqrt_size) % sqrt_size, 0,
                             col_comm, MPI_STATUS_IGNORE);
    }

    // Сбор всех локальных матриц C
    MPI_Gather(local_C.data(), block_size * block_size, MPI_INT, C.data(), block_size * block_size, MPI_INT, 0, grid_comm);

    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
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
            fox_algorithm(A, B, C, n, mpi_rank, mpi_size);
            auto end = chrono::high_resolution_clock::now();

            if (mpi_rank == 0) {
                double duration = chrono::duration<double>(end - start).count();
                total_time += duration;
                //cout << "Size " << n << " Attempt " << attempt + 1 
                //     << ": Time elapsed: " << duration << " seconds\n";
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