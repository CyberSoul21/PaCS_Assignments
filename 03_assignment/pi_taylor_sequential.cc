#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tgmath.h>
#include <chrono>
// Allow to change the floating point type
using my_float = long double;

my_float pi_taylor(size_t steps) {

    my_float sum = 0.0f;
    my_float num = -1;
    //my_float r = 0.0f;
    //(void)steps; // ensure no warnings with unused variable
    for(size_t n = 0; n <= steps; n++)
    {
        num *= -1;
        //r = num/(2*n+1);
        //std::cout<<"r: "<<r<<std::endl;
        sum += num/(2*n+1);
    }
    return 4.0*sum;
}

int main(int argc, const char *argv[]) {

    // read the number of steps from the command line
    if (argc != 2) {
        std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
        exit(1);

    }

    size_t steps = std::stoll(argv[1]);

    auto t0 = std::chrono::steady_clock::now();
    auto pi = pi_taylor(steps);
    auto t1 = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> ms = t1 - t0;
    std::cout << "Time: " << ms.count() << " ms\n";

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<my_float>::digits10 + 1)
        << pi << std::endl;
}
