/* Assignemnt 1
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/

#include <iostream>
#include "/home/javier/eigen/Eigen/Dense"
#include <ctime>
#include <iomanip>

using Eigen::MatrixXi;
using namespace std;

int main() {
    clock_t time;
    time = clock();
    int n = 2000; //matrix dimension
    MatrixXi matrix_1 = MatrixXi::Random(n, n);
    MatrixXi matrix_2 = MatrixXi::Random(n, n);
    MatrixXi matrix_r = matrix_1 * matrix_2;

    time = clock() - time;
    cout << "Seconds: " << fixed << setprecision(4)
        << ((float)time)/CLOCKS_PER_SEC << endl;
}