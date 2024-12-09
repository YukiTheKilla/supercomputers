#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 5.exe 5.cpp
// 5.exe

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
vector<vector<int>> generate_lower_triangular_matrix(int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-1000, 1000);

    vector<vector<int>> matrix(size, vector<int>(size, 0));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= i; ++j) { // Только элементы ниже главной диагонали (включая диагональ)
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

int main() {
    cout << "Task 5: Find the maximum of the minimums of matrix rows" << "\n";

    const int num_runs = 10; // Количество прогонов
    const int data_sizes[] = {500, 1000, 5000}; // Размеры матриц
    const int thread_counts[] = {1, 2, 4, 8}; // Количество потоков    
    
    for (int data_size : data_sizes) {
        cout << "Matrix size: " << data_size << "x" << data_size << endl;

        for (const string& schedule_type : {"static", "dynamic", "guided"}) {
            cout << "Schedule type: " << schedule_type << endl;

            for (int num_threads : thread_counts) {
                omp_set_num_threads(num_threads);

                double total_omp_duration = 0.0;

                for (int run = 0; run < num_runs; ++run) {
                    auto matrix = generate_lower_triangular_matrix(data_size);

                    int max_of_mins = INT_MIN;

                    auto omp_start = chrono::high_resolution_clock::now();

                    #pragma omp parallel
                    {
                        int local_max = INT_MIN;
                        
                        if (schedule_type == "static") {
                            #pragma omp for schedule(static)
                            for (int i = 0; i < data_size; ++i) {
                                int row_min = matrix[i][0];
                                for (int j = 1; j < data_size; ++j) {
                                    row_min = min(row_min, matrix[i][j]);
                                }
                                local_max = max(local_max, row_min);
                            }
                        } else if (schedule_type == "dynamic") {
                            #pragma omp for schedule(dynamic)
                            for (int i = 0; i < data_size; ++i) {
                                int row_min = matrix[i][0];
                                for (int j = 1; j < data_size; ++j) {
                                    row_min = min(row_min, matrix[i][j]);
                                }
                                local_max = max(local_max, row_min);
                            }
                        } else if (schedule_type == "guided") {
                            #pragma omp for schedule(guided)
                            for (int i = 0; i < data_size; ++i) {
                                int row_min = matrix[i][0];
                                for (int j = 1; j < data_size; ++j) {
                                    row_min = min(row_min, matrix[i][j]);
                                }
                                local_max = max(local_max, row_min);
                            }
                        }

                        #pragma omp critical
                        {
                            max_of_mins = max(max_of_mins, local_max);
                        }
                    }

                    auto omp_end = chrono::high_resolution_clock::now();
                    total_omp_duration += chrono::duration<double>(omp_end - omp_start).count();
                }

                cout << "Number of threads: " << num_threads << endl;
                cout << "Average time with " << schedule_type << ": " << total_omp_duration / num_runs << " seconds\n";
            }
            cout << endl;
        }
    }
    return 0;
}