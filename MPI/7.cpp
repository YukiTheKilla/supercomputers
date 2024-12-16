#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-100, 100);

    const int num_runs = 10;
    const int data_sizes[] = {100000, 1000000, 10000000, 100000000};

    for (int data_size : data_sizes) {
        if (rank == 0) {
            cout << "Data size: " << data_size << endl;
        }

        double total_time = 0.0;

        for (int run = 0; run < num_runs; ++run) {
            vector<int> vec_a, vec_b;
            if (rank == 0) {
                vec_a.resize(data_size);
                vec_b.resize(data_size);
                for (int i = 0; i < data_size; ++i) {
                    vec_a[i] = dis(gen);
                    vec_b[i] = dis(gen);
                }
            }

            int chunk_size = data_size / size;
            vector<int> local_a(chunk_size);
            vector<int> local_b(chunk_size);

            MPI_Request send_reqs[2 * (size - 1)];
            MPI_Request recv_reqs[2];

            // Процесс 0 отправляет данные
            if (rank == 0) {
                for (int i = 1; i < size; ++i) {
                    MPI_Isend(vec_a.data() + i * chunk_size, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD, &send_reqs[2 * (i - 1)]);
                    MPI_Isend(vec_b.data() + i * chunk_size, chunk_size, MPI_INT, i, 1, MPI_COMM_WORLD, &send_reqs[2 * (i - 1) + 1]);
                }
                local_a.assign(vec_a.begin(), vec_a.begin() + chunk_size);
                local_b.assign(vec_b.begin(), vec_b.begin() + chunk_size);
            } else {
                MPI_Irecv(local_a.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_reqs[0]);
                MPI_Irecv(local_b.data(), chunk_size, MPI_INT, 0, 1, MPI_COMM_WORLD, &recv_reqs[1]);
            }

            // Локальное вычисление
            auto run_start = chrono::high_resolution_clock::now();

            if (rank != 0) {
                MPI_Waitall(2, recv_reqs, MPI_STATUSES_IGNORE);
            }
            long long local_dot_product = 0;
            for (int i = 0; i < chunk_size; ++i) {
                local_dot_product += static_cast<long long>(local_a[i]) * local_b[i];
            }

            long long global_dot_product = 0;

            // Суммирование результатов
            MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            auto run_end = chrono::high_resolution_clock::now();

            if (rank == 0) {
                auto duration = chrono::duration<double>(run_end - run_start).count();
                total_time += duration;
                // cout << "Run " << run + 1 << ": Dot Product = " << global_dot_product << endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            double average_time = total_time / num_runs;
            cout << "Average time: " << average_time << " seconds\n";
        }
    }

    MPI_Finalize();
    return 0;
}