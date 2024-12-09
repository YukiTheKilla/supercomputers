#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
#include <chrono>
#include <random>

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 1.exe 1.cpp
// 1.exe

int main() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-2147483647, 2147483647);
    cout << "Task 1. MinMax" << "\n";
    const int num_runs = 10; // Количество прогонов
    const int data_sizes[] = {100000, 1000000, 10000000, 100000000}; // Размеры данных
    const int thread_counts[] = {1, 2, 4, 8}; // Количество потоков

    for (int data_size : data_sizes) {
        cout << "Data size: " << data_size << endl;
        for (int num_threads : thread_counts) {
            // Устанавливаем количество потоков
            omp_set_num_threads(num_threads);

            double total_omp_duration = 0.0;
            double total_ompr_duration = 0.0;

            // Прогоняем код num_runs раз
            for (int run = 0; run < num_runs; ++run) {
                vector<int> data;
                for (int i = 0; i < data_size; ++i) {
                    data.push_back(dis(gen));
                }

                int min_value, max_value;

                // Инициализация минимального и максимального значений
                min_value = std::numeric_limits<int>::max();
                max_value = std::numeric_limits<int>::min();

                // Без использования редукции
                auto omp_start = chrono::high_resolution_clock::now();
                #pragma omp parallel
                {
                    int local_min = std::numeric_limits<int>::max();
                    int local_max = std::numeric_limits<int>::min();

                    #pragma omp for
                    for (size_t i = 0; i < data.size(); i++) {
                        if (data[i] < local_min) local_min = data[i];
                        if (data[i] > local_max) local_max = data[i];
                    }

                    #pragma omp critical
                    {
                        if (local_min < min_value) min_value = local_min;
                        if (local_max > max_value) max_value = local_max;
                    }
                }
                auto omp_end = chrono::high_resolution_clock::now();
                total_omp_duration += chrono::duration<double>(omp_end - omp_start).count();

                // С использованием редукции
                auto ompr_start = chrono::high_resolution_clock::now();
                #pragma omp parallel for reduction(min:min_value) reduction(max:max_value)
                for (size_t i = 0; i < data.size(); i++) {
                    if (data[i] < min_value) min_value = data[i];
                    if (data[i] > max_value) max_value = data[i];
                }
                auto ompr_end = chrono::high_resolution_clock::now();
                total_ompr_duration += chrono::duration<double>(ompr_end - ompr_start).count();
            }

            cout << "Number of threads: " << num_threads << endl;
            cout << "Average time without reduction: " << total_omp_duration / num_runs << " seconds\n";
            cout << "Average time with reduction: " << total_ompr_duration / num_runs << " seconds\n";
        }
        cout << endl;
    }

    return 0;
}
