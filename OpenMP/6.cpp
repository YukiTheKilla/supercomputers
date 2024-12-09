#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>
#include <thread> // Для имитации задержки

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 6.exe 6.cpp
// 6.exe

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2); // Рекурсивное вычисление чисел Фибоначчи
}

// Имитация "тяжелой" функции с неравномерной нагрузкой
void heavy_computation(int iteration) {
    for (volatile int i = 0; i < iteration * 5; ++i) {
        volatile int result = fibonacci(i % 2); // Числа Фибоначчи
    }
}

// Основная функция
int main() {
    cout << "Task 6: Uneven workload in for-loop\n";

    const int num_runs = 10; // Количество прогонов
    const int thread_counts[] = {1,2,4,8}; // Количество потоков
    const int num_iterations[] = {1000, 5000, 10000}; // Разное количество итераций для теста

    for (const string& schedule_type : {"static", "dynamic", "guided"}) {
        cout << "Schedule type: " << schedule_type << endl;

        for (int num_threads : thread_counts) {
            omp_set_num_threads(num_threads);

            for (int iterations : num_iterations) {
                double total_omp_duration = 0.0;

                for (int run = 0; run < num_runs; ++run) {
                    auto omp_start = chrono::high_resolution_clock::now();

                    #pragma omp parallel
                    {
                        if (schedule_type == "static") {
                            #pragma omp for schedule(static)
                            for (int i = 0; i < iterations; ++i) {
                                heavy_computation(i);
                            }
                        }
                        else if (schedule_type == "dynamic") {
                            #pragma omp for schedule(dynamic)
                            for (int i = 0; i < iterations; ++i) {
                                heavy_computation(i);
                            }
                        }
                        else if (schedule_type == "guided") {
                            #pragma omp for schedule(guided)
                            for (int i = 0; i < iterations; ++i) {
                                heavy_computation(i);
                            }
                        }
                    }

                    auto omp_end = chrono::high_resolution_clock::now();
                    total_omp_duration += chrono::duration<double>(omp_end - omp_start).count();
                }

                cout << "Number of threads: " << num_threads << endl;
                cout << "Iterations: " << iterations << endl;
                cout << "Average time with " << schedule_type << ": " << total_omp_duration / num_runs << " seconds\n";
            }
        }
        cout << endl;
    }

    return 0;
}