#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 4.exe 4.cpp
// 4.exe

// Функция для генерации случайных данных
vector<vector<int>> generate_matrix(int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-1000, 1000); // Диапазон чисел для элементов матрицы
    vector<vector<int>> matrix(rows, vector<int>(cols));

    // Заполнение матрицы случайными числами
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

int main() {
    cout << "Task 4: Find the maximum of the minimums of matrix rows" << "\n";

    const int num_runs = 10; // Количество прогонов
    const int data_sizes[] = {500, 1000, 5000, 10000}; // Размеры матриц
    const int thread_counts[] = {1, 2, 4, 8}; // Количество потоков

    for (int data_size : data_sizes) {
        cout << "Matrix size: " << data_size << "x" << data_size << endl;

        for (int num_threads : thread_counts) {
            // Устанавливаем количество потоков
            omp_set_num_threads(num_threads);

            double total_omp_duration = 0.0;

            // Прогоняем код num_runs раз
            for (int run = 0; run < num_runs; ++run) {
                // Генерируем случайную матрицу
                auto matrix = generate_matrix(data_size, data_size);

                int max_of_mins = INT_MIN; // Инициализируем максимальное значение минимальных элементов

                auto omp_start = chrono::high_resolution_clock::now();

                #pragma omp parallel
                {
                    int local_max = INT_MIN;

                    #pragma omp for
                    for (int i = 0; i < data_size; ++i) {
                        int row_min = matrix[i][0];
                        for (int j = 1; j < data_size; ++j) {
                            row_min = min(row_min, matrix[i][j]);
                        }
                        local_max = max(local_max, row_min); // Находим максимум среди минимальных элементов
                    }

                    #pragma omp critical
                    {
                        max_of_mins = max(max_of_mins, local_max); // Обновляем глобальный максимум
                    }
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
