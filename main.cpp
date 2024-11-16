#include <iostream>
#include <omp.h>
#include <vector>
#include <limits>

int main() {
    // Инициализация вектора
    std::vector<int> vec = {10, 20, 30, 5, 15, 40, 25};
    
    // Инициализация минимального значения (для минимизации)
    int min_val = std::numeric_limits<int>::max();
    
    // Параллельный поиск минимального значения
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {
        #pragma omp critical
        {
            if (vec[i] < min_val) {
                min_val = vec[i];
            }
        }
    }
    
    std::cout << "Минимальное значение: " << min_val << std::endl;

    return 0;
}