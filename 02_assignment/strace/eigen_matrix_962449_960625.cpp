/* Assignemnt 2
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/

#include <iostream>
#include <sys/time.h>
// #include "/home/javier/eigen/Eigen/Dense"
#include "D:/eigen/eigen-master/Eigen/Dense"
using Eigen::MatrixXi;

int main() {
    double time1, time2, time3;
    struct timeval timestamp;
    gettimeofday(&timestamp, NULL); //Beginning of memory allocation block
    time1 = (double)timestamp.tv_sec + ((double)timestamp.tv_usec)/1000000;
    
    int n = 100; //matrix dimension
    MatrixXi matrix_1 = MatrixXi::Random(n, n);
    MatrixXi matrix_2 = MatrixXi::Random(n, n);
    

    gettimeofday(&timestamp, NULL); //End of memory allocation block + Beginning of matrix multiplication
    time2 = (double)timestamp.tv_sec + ((double)timestamp.tv_usec)/1000000;   
    
    MatrixXi matrix_r = matrix_1 * matrix_2;

    gettimeofday(&timestamp, NULL); //End of matrix multiplication
    time3 = (double)timestamp.tv_sec + ((double)timestamp.tv_usec)/1000000;

    // We do this last substractions at the end to affect the least possible to the algorithm measures
    cout<<"Declaration and memory allocation:"<<time2-time1<<endl;
    cout<<"Matrix multiplication:"<<time3-time2<<endl;
}