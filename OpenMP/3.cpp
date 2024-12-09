#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>
#include <cmath>

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 3.exe 3.cpp
// 3.exe

double f(double x) {
    return exp(x);
}

int main() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-2147483647, 2147483647);
    cout << "Task 3. Integral calculation (Rectangle method)" << "\n";

    const int num_runs = 10; // Количество прогонов
    const int data_sizes[] = {100000, 1000000, 10000000, 100000000}; // Размеры данных
    const int thread_counts[] = {1, 2, 4, 8}; // Количество потоков

    // Пределы интегрирования
    const double a = 0.0;
    const double b = 1.0;

    for (int data_size : data_sizes) {
        cout << "Data size: " << data_size << endl;

        for (int num_threads : thread_counts) {
            // Устанавливаем количество потоков
            omp_set_num_threads(num_threads);

            double total_omp_duration = 0.0;

            // Прогоняем код num_runs раз
            for (int run = 0; run < num_runs; ++run) {
                // Шаг разбиения
                double h = (b - a) / data_size;

                double integral = 0.0;

                auto omp_start = chrono::high_resolution_clock::now();
                
                #pragma omp parallel
                {
                    double local_integral = 0.0;

                    #pragma omp for
                    for (int i = 0; i < data_size; i++) {
                        double x_i = a + i * h + h / 2.0; // Середина прямоугольника
                        local_integral += f(x_i);
                    }

                    #pragma omp atomic
                    integral += local_integral;  // Используем атомарное добавление
                }

                integral *= h; // Умножаем на шаг

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
