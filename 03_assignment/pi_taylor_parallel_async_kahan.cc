#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <future>

//using my_float = long double;
using my_float = float;

typedef struct {
    size_t large_chunk;
    size_t small_chunk;
    size_t split_item;
} chunk_info;

my_float
pi_taylor_chunk(size_t start_step, size_t stop_step) {

    my_float one = (start_step % 2 == 0) ? 1.0f : -1.0f;
    my_float sum = 0.0f;
    
    // Variable to store the error
    my_float c = 0.0f;


    //(start_step % 2 == 0) ? one = 1 : one = -1; 

    for(size_t i = start_step; i < stop_step; i++){

        my_float y = one / (2*i + 1) - c;
        my_float t = sum + y; 

        c = (t - sum) - y; 
        sum = t;
        one *= -1;
    };
    return sum;
}


constexpr chunk_info
split_evenly(size_t N, size_t threads)
{
    return {N / threads + 1, N / threads, N % threads};
}

std::pair<size_t, size_t>
get_chunk_begin_end(const chunk_info& ci, size_t index)
{
    size_t begin = 0, end = 0;
    if (index < ci.split_item ) {
        begin = index*ci.large_chunk;
        end = begin + ci.large_chunk; // (index + 1) * ci.large_chunk
    } else {
        begin = ci.split_item*ci.large_chunk + (index - ci.split_item) * ci.small_chunk;
        end = begin + ci.small_chunk;
    }
    return std::make_pair(begin, end);
}


std::pair<size_t, size_t>
usage(int argc, const char *argv[]) {
    // read the number of steps from the command line
    if (argc != 3) {
        std::cerr << "Invalid syntax: pi_taylor <steps> <threads>" << std::endl;
        exit(1);
    }

    size_t steps = std::stoll(argv[1]);
    size_t threads = std::stoll(argv[2]);

    if (steps < threads ){
        std::cerr << "The number of steps should be larger than the number of threads" << std::endl;
        exit(1);

    }
    return std::make_pair(steps, threads);
}

int main(int argc, const char *argv[]) {


    auto ret_pair = usage(argc, argv);
    auto steps = ret_pair.first;
    auto threads = ret_pair.second;

    my_float pi = 0.0f;

    // please complete missing parts
    std::vector<std::future<my_float>> futureSums;
    futureSums.reserve(threads);

    auto chunks = split_evenly(steps, threads);

    // launch the work
    auto start = std::chrono::steady_clock::now();  // Start timing

    for (size_t i = 0; i < threads; ++i) {
        auto begin_end = get_chunk_begin_end(chunks, i);
        futureSums.emplace_back(std::async(std::launch::async, pi_taylor_chunk,begin_end.first,begin_end.second));
    }

    // Variable to store the error
    my_float c = 0.0f;

    for(auto &r : futureSums)
    {
        my_float y = r.get() - c;
        my_float t = pi + y; 
        c = (t - pi) - y; 
        pi = t;     
    }

    pi*=4.0f;

    auto end = std::chrono::steady_clock::now();    // End timing

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " ms\n";    

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<my_float>::digits10 + 1)
        << pi << std::endl;
}

