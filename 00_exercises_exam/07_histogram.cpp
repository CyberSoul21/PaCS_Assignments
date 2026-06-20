#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <mutex>
#include <fstream>
#include <algorithm>
#include <atomic>


//std::atomic in C++ is used when multiple threads access the same variable, and at least one thread modifies it.
//It makes operations on that variable thread-safe without using a mutex. example: std::atomic<int> counter = 0;
//protects simple shared values from race conditions without needing

namespace
{
    int vectorSize = 100;
//     std::mutex var_mutex;
//     std::condition_variable cv;
//     std::queue<int> sharedQueue;
//     bool finished = false;
}



template<typename T>
void compute_histogram(const std::vector<T>& array,std::vector<std::atomic<int>>& histogram,const size_t chunk, const size_t extraChunk, const size_t id_thread)
{
    size_t begin = id_thread * chunk + std::min(id_thread, extraChunk);
    size_t size = chunk + (id_thread < extraChunk ? 1 : 0);
    size_t end = begin + size;    


    for(size_t i = begin; i < end; ++i) 
    {
        histogram[array[i]] += 1;
    }
}


int main(int argc, char* argv[])
{

    //0. Creating data:
    
    const size_t m_buckets = 32; /// buckets
    const size_t n_threads = 8;

    std::vector<std::atomic<int>> histogram(m_buckets);

    const size_t N = 1024*8; /// array size
    std::vector<size_t> array(N);

    std::random_device rnd_device;
    std::mt19937 gen(rnd_device());
    std::uniform_int_distribution<> dis(0, m_buckets - 1);

    for (size_t i = 0; i < N; ++i)
    {
        array[i] = dis(gen);
    }   


    //Just for testing:
    // std::vector<size_t> array = {
    //     0, 1, 2, 3, 4,
    //     5, 5, 5,
    //     10, 10,
    //     20, 21, 22,
    //     31, 31
    // };

    // const size_t N = array.size();
    

    //for the fork-joink
    size_t chunk = 0;
    size_t extraChunk = 0;

    
    chunk =  N/n_threads;        
    if ( N%n_threads != 0 ) 
    {
        //This will be added to the size of the last vector/thread
        extraChunk = N%n_threads;       
    }    


    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_threads; i++)
    {
        threads.emplace_back(compute_histogram<size_t>,std::cref(array),std::ref(histogram),chunk,extraChunk,i);
    }
    
    for (auto& t : threads)
    {
        t.join();
    }


    std::cout << "Final histogram array: [";

    for (const auto& value : histogram)
    {
        std::cout << value.load() << " ";
    }

    std::cout <<"]"<< std::endl;  




    return 0;
}