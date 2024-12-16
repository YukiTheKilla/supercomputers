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
            vector<int> data(data_size); // Инициализируем вектор на каждом процессе
            if (rank == 0) {
                // Генерация данных на процессе 0
                for (int i = 0; i < data_size; ++i) {
                    data[i] = dis(gen);
                }
            }

            // Записываем время начала передачи
            auto run_start = chrono::high_resolution_clock::now();

            if (rank == 0) {
                // Процесс 0 отправляет данные процессу 1
                MPI_Send(data.data(), data_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
                // Процесс 0 получает данные обратно от процесса 1
                MPI_Recv(data.data(), data_size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                // Процесс 1 получает данные от процесса 0
                MPI_Recv(data.data(), data_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Процесс 1 отправляет данные обратно процессу 0
                MPI_Send(data.data(), data_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }

            // Записываем время окончания выполнения
            auto run_end = chrono::high_resolution_clock::now();

            // Процесс 0 аккумулирует время выполнения
            if (rank == 0) {
                auto duration = chrono::duration<double>(run_end - run_start).count();
                total_time += duration;
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