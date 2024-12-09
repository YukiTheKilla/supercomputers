#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <optional>
#include <queue>
#include <mutex>
#include <condition_variable>
using namespace std;

//cd OpenMP
// g++ -fopenmp -o 8.exe 8.cpp
// 8.exe

template <typename T>
class safe_queue {
private:
    queue<T> queue_;
    mutable mutex mtx_;
    condition_variable cv_;
    bool stop_flag = false;

public:
    
    void push(const T& value) {
        {
            lock_guard<mutex> lock(mtx_);
            queue_.push(value);
        }
        cv_.notify_one(); // notify one waiting thread
    }

    void push(T&& value) {
        {
            lock_guard<mutex> lock(mtx_);
            queue_.push(value);
        }
        cv_.notify_one(); // notify one waiting thread
    }

    // Вытаскиваем последний элемент из очереди. Если пустая очередь то ждем пока появится такой элемент
    optional<T> pop() {
        unique_lock<mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return !queue_.empty() || stop_flag; });

        if (queue_.empty() && stop_flag) {
            return nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    // Остановка очереди
    void stop() {
        {
            lock_guard<mutex> lock(mtx_);
            stop_flag = true;
        }
        cv_.notify_all();
    }
};

void load_vectors(const string& root_filename, safe_queue<pair<pair<int*, int>, pair<int*, int>>>& qs) {
    vector<int> vec1(100);
    vector<int> vec2(100);

    for (int i = 0; i < 100; ++i) {
        vec1[i] = i + 1;      
        vec2[i] = (i + 1) * 2;
        //cout << "vec1[" << i << "] = " << vec1[i] << ", vec2[" << i << "] = " << vec2[i] << endl;
    }

    qs.push({{vec1.data(), static_cast<int>(vec1.size())}, {vec2.data(), static_cast<int>(vec2.size())}});

    qs.stop();
}   

// section
pair<int, double> scalar_product(const string& root_filename) {
    int sc_product = 0;
    safe_queue<pair<pair<int*, int>, pair<int*, int>>> qs;
    
    double start_time = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            load_vectors(root_filename, qs);
        }

        #pragma omp section
        {
            while (true) {
                auto data = qs.pop();
                if (!data.has_value()) {
                    break; // no data
                }

                const auto &[v1, v2] = *data;
                int local_product = 0;

                for (size_t i = 0; i < v1.second; ++i) {
                    local_product += v1.first[i] * v2.first[i];
                }

                #pragma omp atomic
                sc_product += local_product;
            }
        }
    }
    double end_time = omp_get_wtime();

    return {sc_product, end_time - start_time};
}

// simple force
pair<int, double> force_scalar_product(const string& root_filename) {
    int result = 0;
    safe_queue<pair<pair<int*, int>, pair<int*, int>>> qs;

    double start_time = omp_get_wtime();

    load_vectors(root_filename, qs);

    auto data = qs.pop();
    if (data.has_value()) {
        const auto& [v1, v2] = *data;
        for (size_t i = 0; i < v1.second; i++) {
            result += v1.first[i] * v2.first[i];
        }
    }

    return {result, omp_get_wtime() - start_time};
}

int main() {
    string root_filename = "vectors.txt";

    double total_exec_time_parallel = 0;
    double total_exec_time_force = 0;
    int iterations = 10000;

    for (int i = 0; i < iterations; ++i) {
        auto [sc_product, exec_time] = scalar_product(root_filename);
        total_exec_time_parallel += exec_time;
        //cout << "Iteration " << i + 1 << ": Scalar product (parallel sections): " << sc_product << ", Execution time (parallel sections): " << exec_time << " seconds" << endl;

        auto [force_product, force_exec_time] = force_scalar_product(root_filename);
        total_exec_time_force += force_exec_time;
        //cout << "Iteration " << i + 1 << ": Scalar product (force computation): " << force_product << ", Execution time (force computation): " << force_exec_time << " seconds" << endl;
    }

    cout << "\nAverage execution time (parallel sections): " << total_exec_time_parallel / iterations << " seconds" << endl;
    cout << "Average execution time (force computation): " << total_exec_time_force / iterations << " seconds" << endl;

    return 0;
}
