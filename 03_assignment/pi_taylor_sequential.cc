#include <iomanip>
#include <iostream>
#include <limits>

// Allow to change the floating point type
using my_float = long double;

my_float pi_taylor(size_t steps) {

    (void)steps; // ensure no warnings with unused variable 
    my_float pi = 0.0f;
    for(size_t i = 0; i < steps; i++){
        pi += 4* ((pow(-1,i)) / (2*i + 1));
    }
    return pi;
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
