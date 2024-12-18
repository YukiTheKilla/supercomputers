#include <cstdlib> // Для system()
#include <sstream> // Для std::ostringstream
#include <iostream>
/*  Я курю бамбук    */
int ex1(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 8; // Максимальное количество узлов
    const int k = 8;
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

int ex2(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 8; // Максимальное количество узлов
    const int k = 8;
    std::cout << "Task 2. Scalar" << "\n";
    for (int n = min; n <= max; n=n*2) {
        std::ostringstream command;

        command << "srun -p gnu --ntasks="<< n << " --nodes=" << n << " 2";

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

int ex3(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 8; // Максимальное количество узлов
    const int k = 8;
    std::cout << "Task 3. Connections" << "\n";
    for (int n = min; n <= max; n=n*2) {
        std::ostringstream command;

        command << "srun -p gnu --ntasks="<< n*2 << " --nodes=" << n << " 3";

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

int ex4_1(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 4; // Максимальное количество узлов
    const int k = 8;
    std::cout << "Task 4_1. Matrix Multiplication" << "\n";
    for (int n = min; n <= max; n=n*4) {
        std::ostringstream command;

        command << "srun -p gnu" << " --nodes=" << n << " 4_1";

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

int ex4_2(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 4; // Максимальное количество узлов
    const int k = 8;
    std::cout << "Task 4_2. MatrixMultiplication" << "\n";
    for (int n = min; n <= max; n=n*4) {
        std::ostringstream command;

        command << "srun -p gnu" << " --nodes=" << n << " 4_2";

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

int ex5(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 8; // Максимальное количество узлов
    const int k = 8;
    std::cout << "Task 5. Communication time" << "\n";
    for (int n = min; n <= max; n=n*2) {
        std::ostringstream command;

        command << "srun -p gnu --ntasks="<< n << " --nodes=" << n << " 5";

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

int ex6(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 8; // Максимальное количество узлов
    const int k = 8;
    std::cout << "Task 6. Operation types" << "\n";
    for (int n = min; n <= max; n=n*2) {
        std::ostringstream command;

        command << "srun -p gnu" << " --nodes=" << n << " 6";

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

int ex7(){
    const int min = 1;  // Минимальное количество узлов
    const int max = 8; // Максимальное количество узлов
    const int k = 8;
    std::cout << "Task 7. Locker" << "\n";
    for (int n = min; n <= max; n=n*2) {
        std::ostringstream command;

        command << "srun -p gnu --ntasks="<< n << " --nodes=" << n << " 7";

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
    return ex7();
}