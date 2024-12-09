#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
#include <chrono>
#include <random>
#include <mutex>

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 7.exe 7.cpp
// 7.exe


int main() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-2147483647, 2147483647);
    cout << "Task 7. atom,critical,mutex,reduction" << "\n";
    const int num_runs = 10;
    const int data_sizes[] = {10000, 100000, 1000000, 10000000};
    const int thread_counts[] = {1, 2, 4, 8};

    for (int data_size : data_sizes) {
        cout << "Data size: " << data_size << endl;

        for (int num_threads : thread_counts) {
            cout << "Number of threads: " << num_threads << endl;
            omp_set_num_threads(num_threads);

            double total_atomic_duration = 0.0;
            double total_critical_duration = 0.0;
            double total_mutex_duration = 0.0;
            double total_reduction_duration = 0.0;

            for (int run = 0; run < num_runs; ++run) {
                vector<int> data;
                for (int i = 0; i < data_size; ++i) {
                    data.push_back(dis(gen));
                }

                int min_value, max_value;

                // Атомарные операции
                min_value = std::numeric_limits<int>::max();
                max_value = std::numeric_limits<int>::min();
                auto atomic_start = chrono::high_resolution_clock::now();
                #pragma omp parallel
                {
                    #pragma omp for
                    for (size_t i = 0; i < data.size(); i++) {
                        if (data[i] < min_value) {
                            #pragma omp atomic write
                            min_value = data[i];
                        }
                        if (data[i] > max_value) {
                            #pragma omp atomic write
                            max_value = data[i];
                        }
                    }
                }
                auto atomic_end = chrono::high_resolution_clock::now();
                total_atomic_duration += chrono::duration<double>(atomic_end - atomic_start).count();

                // Критическая секция
                min_value = std::numeric_limits<int>::max();
                max_value = std::numeric_limits<int>::min();
                auto critical_start = chrono::high_resolution_clock::now();
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
                auto critical_end = chrono::high_resolution_clock::now();
                total_critical_duration += chrono::duration<double>(critical_end - critical_start).count();

                // С замками
                std::mutex min_max_mutex;
                min_value = std::numeric_limits<int>::max();
                max_value = std::numeric_limits<int>::min();
                auto mutex_start = chrono::high_resolution_clock::now();
                #pragma omp parallel
                {
                    int local_min = std::numeric_limits<int>::max();
                    int local_max = std::numeric_limits<int>::min();

                    #pragma omp for
                    for (size_t i = 0; i < data.size(); i++) {
                        if (data[i] < local_min) local_min = data[i];
                        if (data[i] > local_max) local_max = data[i];
                    }

                    // Защищаем доступ к глобальным переменным через мьютекс
                    {
                        std::lock_guard<std::mutex> lock(min_max_mutex);
                        if (local_min < min_value) min_value = local_min;
                        if (local_max > max_value) max_value = local_max;
                    }
                }
                auto mutex_end = chrono::high_resolution_clock::now();
                total_mutex_duration += chrono::duration<double>(mutex_end - mutex_start).count();

                // Редукция с использованием OpenMP
                min_value = std::numeric_limits<int>::max();
                max_value = std::numeric_limits<int>::min();
                auto reduction_start = chrono::high_resolution_clock::now();
                #pragma omp parallel for reduction(min:min_value) reduction(max:max_value)
                for (size_t i = 0; i < data.size(); i++) {
                    if (data[i] < min_value) min_value = data[i];
                    if (data[i] > max_value) max_value = data[i];
                }
                auto reduction_end = chrono::high_resolution_clock::now();
                total_reduction_duration += chrono::duration<double>(reduction_end - reduction_start).count();
            }

            // Выводим результаты для всех методов
            cout << "Average time using atomic operations: " << total_atomic_duration / num_runs << " seconds\n";
            cout << "Average time using critical section: " << total_critical_duration / num_runs << " seconds\n";
            cout << "Average time using mutex: " << total_mutex_duration / num_runs << " seconds\n";
            cout << "Average time using reduction: " << total_reduction_duration / num_runs << " seconds\n";
        }
        cout << endl;
    }

    return 0;
}
