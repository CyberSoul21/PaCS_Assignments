#include <iomanip>
#include <iostream>
#include <limits>
#include <cmath>
#include <string>
// Allow to change the floating point type
using my_float = long double;

my_float pi_taylor(size_t steps) {

    (void)steps; // ensure no warnings with unused variable 
    my_float pi = 0.0f;
    int one = 1;
    for(size_t i = 0; i < steps; i++){
        pi += one / (2*i + 1);
        one = -one;
    }
    return pi*4;
}

int main(int argc, const char *argv[]) {

    // read the number of steps from the command line
    if (argc != 2) {
        std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
        exit(1);

    }

    size_t steps = std::stoll(argv[1]);
    auto pi = pi_taylor(steps);

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<my_float>::digits10 + 1)
        << pi << std::endl;
}
