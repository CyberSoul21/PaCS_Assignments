/* Assignemnt 1
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/

#include <iostream>
#include <random>
#include <ctime>
#include <iomanip>

using namespace std;

int main() {

    clock_t time;
    time = clock();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};

    int n =  2000; //matrix dimension

    double* matrix_1 = new double[n * n];
    double* matrix_2 = new double[n * n];
    double* matrix_r = new double[n * n];


    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            *(matrix_1 + i * n + j) = dis(gen);
            *(matrix_2 + i * n + j) = dis(gen);
            *(matrix_r + i * n + j) = 0;
        }
    }
    
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int k = 0; k < n; k++){
                *(matrix_r + i * n + j) += *(matrix_1 + i * n + k) * *(matrix_2 + k * n + j);
            }
        }
    }

    delete [] matrix_r, matrix_1, matrix_2;
    time = clock() - time;
    cout << "Seconds: " << fixed << setprecision(4)
        << ((float)time)/CLOCKS_PER_SEC << endl;
}