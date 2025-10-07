/* Assignemnt 1
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/

#include <iostream>
#include <random>
#include <sys/time.h>

using namespace std;

int main() {
    struct timeval timestamp;
    gettimeofday(&timestamp, NULL);
    cout << "Start declaration and mem allocation. "<< "Seconds: " << timestamp.tv_sec << endl
        << "Microseconds: " << timestamp.tv_usec << endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};

    int n =  100; //matrix dimension

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
    
    gettimeofday(&timestamp, NULL);
    cout << "End declaration, start multiplication. "<< "Seconds: " << timestamp.tv_sec << endl
        << "Microseconds: " << timestamp.tv_usec << endl;
    
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int k = 0; k < n; k++){
                *(matrix_r + i * n + j) += *(matrix_1 + i * n + k) * *(matrix_2 + k * n + j);
            }
        }
    }

    gettimeofday(&timestamp, NULL);
    cout << "End multiplication. "<< "Seconds: " << timestamp.tv_sec << endl
        << "Microseconds: " << timestamp.tv_usec << endl;

        delete [] matrix_r, matrix_1, matrix_2;

    gettimeofday(&timestamp, NULL);
    cout << "Last delete dynamic memory. "<< "Seconds: " << timestamp.tv_sec << endl
        << "Microseconds: " << timestamp.tv_usec << endl;

}