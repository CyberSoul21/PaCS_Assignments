#include <iostream>
#include <random>

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

    int n =  0; //matrix dimension

    cout<<"Type N: "<<endl;
    cin>>n;

    //create array
    double* matrix_1 = new double[n * n];
    double* matrix_2 = new double[n * n];


    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            *(matrix_1 + i * n + j) = dis(gen);
            cout<<" "<<*(matrix_1 + i * n + j);
        }
        cout<<" "<<endl;
    }

}