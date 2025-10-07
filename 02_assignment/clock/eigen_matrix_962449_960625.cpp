/* Assignemnt 1
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/

#include <iostream>
#include "/home/javier/eigen/Eigen/Dense"
using Eigen::MatrixXi;

int main() {
    int n = 100; //matrix dimension
    MatrixXi matrix_1 = MatrixXi::Random(n, n);
    MatrixXi matrix_2 = MatrixXi::Random(n, n);
    MatrixXi matrix_r = matrix_1 * matrix_2;
}