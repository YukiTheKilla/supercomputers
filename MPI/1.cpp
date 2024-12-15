#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <random>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Инициализация MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем идентификатор процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получаем количество процессов

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-2147483647, 2147483647);

    const int num_runs = 10;
    const int data_sizes[] = {100000, 1000000, 10000000, 100000000}; // Размеры данных

    // Разделим работу по процессам
    for (int data_size : data_sizes) {
        if (rank == 0) {
            cout << "Data size: " << data_size << endl;
        }

        double total_time = 0.0; // Для накопления времени всех прогонов

        for (int run = 0; run < num_runs; ++run) {
            vector<int> data;
            if (rank == 0) {
                // Генерируем данные на процессе 0
                for (int i = 0; i < data_size; ++i) {
                    data.push_back(dis(gen));
                }
            }

            // Отправляем данные другим процессам
            int chunk_size = data_size / size;
            vector<int> local_data(chunk_size);

            // Распределяем данные между процессами
            MPI_Scatter(data.data(), chunk_size, MPI_INT, local_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

            // Записываем время начала выполнения
            auto run_start = chrono::high_resolution_clock::now();

            // Инициализация минимального и максимального значений
            int local_min = std::numeric_limits<int>::max();
            int local_max = std::numeric_limits<int>::min();

            // Вычисление min и max для локальных данных
            for (int i = 0; i < chunk_size; ++i) {
                if (local_data[i] < local_min) local_min = local_data[i];
                if (local_data[i] > local_max) local_max = local_data[i];
            }

            // Сборка результатов с всех процессов
            int global_min, global_max;
            MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

            // Записываем время окончания выполнения
            auto run_end = chrono::high_resolution_clock::now();

            // Процесс 0 аккумулирует время выполнения
            if (rank == 0) {
                auto duration = chrono::duration<double>(run_end - run_start).count();
                total_time += duration;
            }

            // Синхронизация всех процессов
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Вычисление среднего времени
        if (rank == 0) {
            double average_time = total_time / num_runs;
            cout << "Average time: " << average_time << " seconds\n";
        }
    }

    MPI_Finalize(); // Завершаем работу MPI
    return 0;
}
