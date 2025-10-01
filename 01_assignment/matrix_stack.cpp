#include <iostream>
#include <random>
#include <sys/resource.h>
using namespace std;

//NOTES:
//notes about assignment
//command time to measure time
//Report, be specific how you made the execution, number of repeats, size matrix...etc

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};
    //auto pseudo_random_float_value = dis(gen);

    int n =  100; //matrix dimension //up to 592
    double a[n][n];
    double b[n][n];
    double c[n][n];

    //create array
    double* matrix_1 = &a[0][0];
    double* matrix_2 = &b[0][0];
    double* matrix_r = &c[0][0];

    struct rlimit limit;    

    //cout<<"Stack size: "<<getrlimit(RLIMIT_STACK,&limit)<<endl;
    getrlimit(RLIMIT_STACK,&limit);
    printf("\nStack limit: %ld and %ld max \n",limit.rlim_cur,limit.rlim_max);

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
}