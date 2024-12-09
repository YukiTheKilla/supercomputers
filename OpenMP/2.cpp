#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 2.exe 2.cpp
// 2.exe

int main() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-2147483647, 2147483647);
    cout << "Task 2. Vector scalar" << "\n";

    const int num_runs = 10; // Количество прогонов
    const int data_sizes[] = {100000, 1000000, 10000000, 100000000}; // Размеры данных
    const int thread_counts[] = {1, 2, 4, 8}; // Количество потоков

    for (int data_size : data_sizes) {
        cout << "Data size: " << data_size << endl;

        for (int num_threads : thread_counts) {
            // Устанавливаем количество потоков
            omp_set_num_threads(num_threads);

            double total_omp_duration = 0.0;

            // Прогоняем код num_runs раз
            for (int run = 0; run < num_runs; ++run) {
                vector<int> vector_a(data_size);
                vector<int> vector_b(data_size);
                
                // Заполняем векторы случайными числами
                for (int i = 0; i < data_size; ++i) {
                    vector_a[i] = dis(gen);
                    vector_b[i] = dis(gen);
                }

                long long scalar_product = 0;

                // Без использования редукции
                auto omp_start = chrono::high_resolution_clock::now();
                #pragma omp parallel
                {
                    long long local_product = 0;

                    #pragma omp for
                    for (size_t i = 0; i < data_size; i++) {
                        local_product += static_cast<long long>(vector_a[i]) * vector_b[i];
                    }

                    #pragma omp atomic
                    scalar_product += local_product;  // Используем атомарное добавление
                }
                auto omp_end = chrono::high_resolution_clock::now();
                total_omp_duration += chrono::duration<double>(omp_end - omp_start).count();
            }

            cout << "Number of threads: " << num_threads << endl;
            cout << "Average time: " << total_omp_duration / num_runs << " seconds\n";
        }
        cout << endl;
    }

    return 0;
}
