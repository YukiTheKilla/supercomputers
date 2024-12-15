#include <cstdlib> // Для system()
#include <sstream> // Для std::ostringstream
#include <iostream>
int ex1(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 8; // Максимальное количество узлов

    std::cout << "Task 1. MinMax" << "\n";
    for (int n = min; n <= max; n=n*2) {
        std::ostringstream command;

        command << "srun -p gnu --ntasks="<< n << " --nodes=" << n << " 1";

        std::cout << "Num cores: " << n << std::endl;

        // Выполняем команду
        int ret_code = system(command.str().c_str());
        if (ret_code != 0) {
            std::cerr << "Error: Command failed with return code " << ret_code << std::endl;
            return ret_code;
        }
        }
        return 0;
}

int main() {
    return ex1();
}












