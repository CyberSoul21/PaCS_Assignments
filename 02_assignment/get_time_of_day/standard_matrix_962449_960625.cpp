/* Assignemnt 2
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/

#include <iostream>
#include <random>
#include <sys/time.h>

using namespace std;

int main() {
    double time1, time2, time3, time4;
    struct timeval timestamp;
    gettimeofday(&timestamp, NULL); //Beginning of memory allocation block
    time1 = (double)timestamp.tv_sec + ((double)timestamp.tv_usec)/1000000; // Justificar que este tipo de mediciones "contaminan" el tiempo real ya que añaden tiempo de ejeución
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};

    int n =  500; //matrix dimension

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
    
    gettimeofday(&timestamp, NULL); //End of memory allocation block + Beginning of matrix multiplication
    time2 = (double)timestamp.tv_sec + ((double)timestamp.tv_usec)/1000000;    
    
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int k = 0; k < n; k++){
                *(matrix_r + i * n + j) += *(matrix_1 + i * n + k) * *(matrix_2 + k * n + j);
            }
        }
    }

    gettimeofday(&timestamp, NULL); //End of matrix multiplication
    time3 = (double)timestamp.tv_sec + ((double)timestamp.tv_usec)/1000000;

    // We consider this last delete sentence as part of the memory allocation block.
    delete [] matrix_r, matrix_1, matrix_2;

    gettimeofday(&timestamp, NULL);
    time4 = (double)timestamp.tv_sec + ((double)timestamp.tv_usec)/1000000;
    
    // We do this last substractions at the end to affect the least possible to the algorithm measures
    cout<<"Declaration, memory allocation and delete:"<<(time2-time1)+(time4-time3)<<endl;
    cout<<"Matrix multiplication:"<<time3-time2<<endl;

}