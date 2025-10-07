/* Assignemnt 1
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/

#include <iostream>
#include <sys/time.h>
// #include "/home/javier/eigen/Eigen/Dense"
#include "D:/eigen/eigen-master/Eigen/Dense"
using Eigen::MatrixXi;

int main() {
    struct timeval timestamp;
    
    gettimeofday(&timestamp, NULL);
    cout << "Start declaration and mem allocation. "<< "Seconds: " << timestamp.tv_sec << endl
        << "Microseconds: " << timestamp.tv_usec << endl;
    
        int n = 100; //matrix dimension
    MatrixXi matrix_1 = MatrixXi::Random(n, n);
    MatrixXi matrix_2 = MatrixXi::Random(n, n);
    

    gettimeofday(&timestamp, NULL);
    cout << "End declaration, start multiplication. "<< "Seconds: " << timestamp.tv_sec << endl
        << "Microseconds: " << timestamp.tv_usec << endl;
    
    MatrixXi matrix_r = matrix_1 * matrix_2;

    gettimeofday(&timestamp, NULL);
    cout << "End multiplication. "<< "Seconds: " << timestamp.tv_sec << endl
        << "Microseconds: " << timestamp.tv_usec << endl
}