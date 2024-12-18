#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Инициализация MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем идентификатор процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получаем количество процессов

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-100, 100); // Для удобства взяты меньшие числа

    const int num_runs = 10;
    const int data_sizes[] = {100000, 1000000, 10000000, 100000000}; // Размеры данных

    for (int data_size : data_sizes) {
        if (rank == 0) {
            cout << "Data size: " << data_size << endl;
        }

        double total_time = 0.0;

        for (int run = 0; run < num_runs; ++run) {
            vector<int> vec_a, vec_b;
            if (rank == 0) {
                // Генерация векторов на процессе 0
                vec_a.resize(data_size);
                vec_b.resize(data_size);
                for (int i = 0; i < data_size; ++i) {
                    vec_a[i] = dis(gen);
                    vec_b[i] = dis(gen);
                }
            }

            // Отправка данных другим процессам
            int chunk_size = data_size / size;
            vector<int> local_a(chunk_size);
            vector<int> local_b(chunk_size);

            vector<MPI_Request> send_requests(2 * (size - 1));
            if (rank == 0) {
                for (int i = 1; i < size; ++i) {
                    MPI_Isend(vec_a.data() + i * chunk_size, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD, &send_requests[2 * (i - 1)]);
                    MPI_Isend(vec_b.data() + i * chunk_size, chunk_size, MPI_INT, i, 1, MPI_COMM_WORLD, &send_requests[2 * (i - 1) + 1]);
                }
            } else {
                MPI_Request recv_requests[2];
                MPI_Irecv(local_a.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_requests[0]);
                MPI_Irecv(local_b.data(), chunk_size, MPI_INT, 0, 1, MPI_COMM_WORLD, &recv_requests[1]);
                MPI_Waitall(2, recv_requests, MPI_STATUSES_IGNORE);
            }

            if (rank == 0) {
                copy(vec_a.begin(), vec_a.begin() + chunk_size, local_a.begin());
                copy(vec_b.begin(), vec_b.begin() + chunk_size, local_b.begin());
            }

            // Записываем время начала выполнения
            auto run_start = chrono::high_resolution_clock::now();

            // Локальное вычисление скалярного произведения
            long long local_dot_product = 0;
            for (int i = 0; i < chunk_size; ++i) {
                local_dot_product += static_cast<long long>(local_a[i]) * local_b[i];
            }

            // Суммирование результатов
            long long global_dot_product = 0;
            MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            // Записываем время окончания выполнения
            auto run_end = chrono::high_resolution_clock::now();

            // Процесс 0 аккумулирует время выполнения
            if (rank == 0) {
                auto duration = chrono::duration<double>(run_end - run_start).count();
                total_time += duration;
                cout << "Run " << run + 1 << ": Dot Product = " << global_dot_product << endl;
            }

            // Завершение отправки на процессе 0
            if (rank == 0) {
                MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
            }

            // Синхронизация процессов
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Среднее время выполнения
        if (rank == 0) {
            double average_time = total_time / num_runs;
            cout << "Average time: " << average_time << " seconds\n";
        }
    }

    MPI_Finalize(); // Завершение работы MPI
    return 0;
}