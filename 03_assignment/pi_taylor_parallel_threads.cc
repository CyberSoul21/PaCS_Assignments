#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using my_float = long double;

void
pi_taylor_chunk(std::vector<my_float> &output,
        size_t thread_id, size_t start_step, size_t stop_step) {
    
    //(void) thread_id;
    my_float one=-1;
    my_float sum = 0.0f;
    
    (start_step % 2 == 0)?one = -1:one = 1;

    for(size_t i = start_step; i < stop_step; i++){
        one *= -1;
        //output[i] = one / (2*i + 1);
        sum += one / (2*i + 1);
        //std::cout<<"r: "<<one / (2*i + 1)<<std::endl;
        //std::cout<<"one: "<<one<<std::endl;
    };
    output[thread_id] = sum;
}

typedef struct {
    size_t large_chunk;
    size_t small_chunk;
    size_t split_item;
} chunk_info;

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



/*
//JAvier //just for testing
std::pair<size_t, size_t>
get_chunk_begin_end(const chunk_info& ci, size_t index)
{

    size_t begin = 0, end = 0;

    if (index == 0)
    {
        begin = 0;
    }
    else
    {
       begin = ci.small_chunk*((index + 1)- 1) + 1;
    }
    end = ci.small_chunk*(index + 1);

 
    return std::make_pair(begin, end);
}
*/

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

    my_float pi=0.0f;
    //size_t pi=0.0f;
    std::vector<my_float> partial_results(threads, 0.0f);
    std::vector<std::thread> myThreads;
    myThreads.reserve(threads);

    // please complete missing parts
    auto chunks = split_evenly(steps, threads);
    //std::cout<<"Large chunks: "<<chunks.large_chunk<<"small chunks: "<<chunks.small_chunk<<" partial_results size: "<<partial_results.size()<<std::endl;
    // ToDo : run several times and check median and deviation

    // launch the work
    auto start = std::chrono::steady_clock::now();  // Start timing

    for(size_t i = 0; i < threads; ++i) {
        auto begin_end = get_chunk_begin_end(chunks, i);
        //std::cout<<"Begin: "<<begin_end.first<<" End: "<<begin_end.second<<std::endl;
        myThreads.push_back(std::thread(pi_taylor_chunk, std::ref(partial_results), i, begin_end.first,begin_end.second));
    }

    for(auto& t: myThreads) {
        t.join();
    }

    
    for(auto r: partial_results) {
        //std::cout<<"r: "<<r<<std::endl;
        pi+=r;
    }

    pi*=4.0f;

    auto end = std::chrono::steady_clock::now();    // End timing

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " ms\n";




    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
        << pi << std::endl;
}

