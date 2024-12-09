#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <climits>
#include <algorithm>

using namespace std;

//cd OpenMP
// g++ -fopenmp -o 9.exe 9.cpp
// 9.exe

// Функция генерации случайной матрицы
vector<vector<int>> generate_matrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 1000;
        }
    }
    return matrix;
}

// Без вложенного параллелизма
double run_no_nested_parallel(int data_size, int num_threads, int num_runs) {
    double total_duration = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        auto matrix = generate_matrix(data_size, data_size);

        int max_of_mins = INT_MIN;

        omp_set_num_threads(num_threads);

        auto omp_start = chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(max:max_of_mins)
        for (int i = 0; i < data_size; ++i) {
            int row_min = matrix[i][0];
            for (int j = 1; j < data_size; ++j) {
                row_min = min(row_min, matrix[i][j]);
            }
            max_of_mins = max(max_of_mins, row_min);
        }

        auto omp_end = chrono::high_resolution_clock::now();
        total_duration += chrono::duration<double>(omp_end - omp_start).count();
    }

    return total_duration / num_runs;
}

// С вложенным параллелизмом
double run_with_nested_parallel(int data_size, int num_threads, int num_runs) {
    double total_duration = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        auto matrix = generate_matrix(data_size, data_size);

        int max_of_mins = INT_MIN;

        omp_set_num_threads(num_threads);

        auto omp_start = chrono::high_resolution_clock::now();

        #pragma omp parallel
        {
            int local_max = INT_MIN;

            #pragma omp for
            for (int i = 0; i < data_size; ++i) {
                int row_min = matrix[i][0];
                // Параллелим внутренний цикл
                #pragma omp parallel for reduction(min:row_min)
                for (int j = 1; j < data_size; ++j) {
                    row_min = min(row_min, matrix[i][j]);
                }
                local_max = max(local_max, row_min);
            }

            #pragma omp critical
            {
                max_of_mins = max(max_of_mins, local_max);
            }
        }

        auto omp_end = chrono::high_resolution_clock::now();
        total_duration += chrono::duration<double>(omp_end - omp_start).count();
    }

    return total_duration / num_runs;
}

int main() {
    cout << "Task 9: Nested parallel" << "\n";

    const int num_runs = 10;
    const int data_sizes[] = {100, 1000, 10000, 20000};
    const int thread_counts[] = {1, 2, 4, 8};

    for (int data_size : data_sizes) {
        cout << "Matrix size: " << data_size << "x" << data_size << endl;

        for (int num_threads : thread_counts) {
            cout << "Number of threads: " << num_threads << endl;

            // Без вложенного параллелизма
            double no_nested_time = run_no_nested_parallel(data_size, num_threads, num_runs);
            cout << "No Nested Parallelism Average Time: " << no_nested_time << " seconds\n";

            // С вложенным параллелизмом
            double with_nested_time = run_with_nested_parallel(data_size, num_threads, num_runs);
            cout << "With Nested Parallelism Average Time: " << with_nested_time << " seconds\n";

            // Сравнение
            cout << "Speedup: " << (no_nested_time / with_nested_time) << "\n\n";
        }
    }

    return 0;
}
