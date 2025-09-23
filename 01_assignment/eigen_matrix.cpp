#include <iostream>
#include "D:\eigen\eigen-master\Eigen\Dense"
//#include "/home/javier/eigen/Eigen/Dense"
using Eigen::MatrixXi;

int main() {
    int n = 1000;
    MatrixXi matrix_1 = MatrixXi::Random(n, n);
    MatrixXi matrix_2 = MatrixXi::Random(n, n);
    MatrixXi matrix_r = matrix_1 * matrix_2;
}