#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

using namespace std;

void test_modes(int mode, int rank, int size, int data_size, int num_runs) {
    vector<int> data(data_size);
    int chunk_size = data_size / size;

    // Создаем вектор данных на процессе 0
    if (rank == 0) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(-100, 100);
        for (int i = 0; i < data_size; ++i) {
            data[i] = dis(gen);
        }
    }

    // Буфер для буферизованного режима
    bool buffer_attached = false;
    if (rank == 0 && mode == 3) {
        int buffer_size = data_size * sizeof(int) + MPI_BSEND_OVERHEAD;
        vector<char> buffer(buffer_size);
        MPI_Buffer_attach(buffer.data(), buffer_size);
        buffer_attached = true;
    }

    double total_time = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        vector<int> local_data(chunk_size);
        auto start_time = chrono::high_resolution_clock::now();

        if (rank == 0) {
            // Передача данных в зависимости от режима
            for (int i = 1; i < size; ++i) {
                if (mode == 1) { // Синхронный режим
                    MPI_Ssend(data.data() + i * chunk_size, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
                } else if (mode == 2) { // Режим "по готовности"
                    MPI_Rsend(data.data() + i * chunk_size, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
                } else if (mode == 3) { // Буферизованный режим
                    MPI_Bsend(data.data() + i * chunk_size, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }
            // Локальные данные процесса 0
            local_data.assign(data.begin(), data.begin() + chunk_size);
        } else {
            // Получение данных
            MPI_Recv(local_data.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Имитация вычислений
        long long local_sum = 0;
        for (int val : local_data) {
            local_sum += val;
        }

        long long global_sum = 0;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        auto end_time = chrono::high_resolution_clock::now();
        double duration = chrono::duration<double>(end_time - start_time).count();

        if (rank == 0) {
            total_time += duration;
        }

        MPI_Barrier(MPI_COMM_WORLD); // Синхронизация процессов
    }

    if (rank == 0) {
        double average_time = total_time / num_runs;
        cout << "Mode " << mode << ", Data Size " << data_size << ": Average Time = " << average_time << " seconds" << endl;
    }

    // Отключение буфера только если он был прикреплен
    if (buffer_attached) {
        void* attached_buffer;
        int buffer_size;
        MPI_Buffer_detach(&attached_buffer, &buffer_size);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int num_runs = 10;
    const int data_sizes[] = {10, 50, 100, 200};

    for (int data_size : data_sizes) {
        for (int mode = 1; mode <= 3; ++mode) {
            test_modes(mode, rank, size, data_size, num_runs);
        }
    }

    MPI_Finalize();
    return 0;
}