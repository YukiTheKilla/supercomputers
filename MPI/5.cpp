#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
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

    // Конфигурация
    const int num_runs = 10;
    const int data_sizes[] = {100000, 1000000, 10000000}; // Размеры данных
    const int compute_time = 100000; // Искусственное время вычислений в микросекундах
    if (rank == 0) {cout << "Compute time: " << compute_time << endl;}
    for (int data_size : data_sizes) {
        if (rank == 0) {
            cout << "Data size: " << data_size << endl;
        }

        double total_compute_time = 0.0, total_comm_time = 0.0;

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

            // Количество данных, передаваемых между процессами
            int chunk_size = data_size / size;
            vector<int> local_a(chunk_size);
            vector<int> local_b(chunk_size);

            // Замер времени коммуникаций (Scatter)
            auto comm_start = chrono::high_resolution_clock::now();
            MPI_Scatter(vec_a.data(), chunk_size, MPI_INT, local_a.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatter(vec_b.data(), chunk_size, MPI_INT, local_b.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
            auto comm_end = chrono::high_resolution_clock::now();
            total_comm_time += chrono::duration<double>(comm_end - comm_start).count();

            // Искусственная нагрузка для вычислений
            auto compute_start = chrono::high_resolution_clock::now();
            this_thread::sleep_for(chrono::microseconds(compute_time)); // Эмулируем вычисления

            // Локальная операция (например, просто сложение случайных чисел)
            long long local_result = 0;
            for (int i = 0; i < chunk_size; ++i) {
                local_result += local_a[i] + local_b[i]; // Искусственная нагрузка
            }
            auto compute_end = chrono::high_resolution_clock::now();
            total_compute_time += chrono::duration<double>(compute_end - compute_start).count();

            // Замер времени коммуникаций (Reduce)
            comm_start = chrono::high_resolution_clock::now();
            long long global_result = 0;
            MPI_Reduce(&local_result, &global_result, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            comm_end = chrono::high_resolution_clock::now();
            total_comm_time += chrono::duration<double>(comm_end - comm_start).count();

            if (rank == 0) {
            //    cout << "Run " << run + 1 << ": Result = " << global_result << endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            cout << "Average compute time: " << total_compute_time / num_runs << " seconds\n";
            cout << "Average communication time: " << total_comm_time / num_runs << " seconds\n";
        }
    }

    MPI_Finalize();
    return 0;
}