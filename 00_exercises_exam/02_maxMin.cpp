#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <mutex>
#include <fstream>

//Rule divide task depending number threads and size of the vector
namespace
{
    std::mutex var_mutex;
}

void getMax(const std::vector<size_t>& v, const size_t begin,const size_t end, size_t& valueMax)
{

    //size_t valueMin = v[begin];
    valueMax = v[begin];
    for(size_t i = begin; i < end; ++i)
    {
        if(v[i] > valueMax)
        {
            valueMax = v[i];
        }
    }

}

void getMin(const std::vector<size_t>& v, const size_t begin,const size_t end, size_t& valueMin)
{
    valueMin = v[begin];
    for(size_t i = begin; i < end; ++i)
    {
        if(v[i] < valueMin)
        {
           valueMin = v[i];
        }
    }
}

void getMaxAndMin(const std::vector<size_t>& v,const size_t chunk, const size_t extraChunk, const size_t id_thread, std::vector<size_t>& max, std::vector<size_t>& min)
{

    size_t begin = id_thread * chunk + std::min(id_thread, extraChunk);
    size_t size = chunk + (id_thread < extraChunk ? 1 : 0);
    size_t end = begin + size;


    size_t valueMin = 0;
    size_t valueMax = 0;

    getMax(v,begin,end,valueMax);
    getMin(v,begin,end,valueMin);

    std::lock_guard<std::mutex> lck(var_mutex);
    {
        min.push_back(valueMin);
        max.push_back(valueMax);
    }
}

int main()
{
    size_t vector_size = 100000000;//200;
    std::vector<size_t> numericVector(vector_size);
    std::vector<size_t> valuesMax;
    std::vector<size_t> valuesMin;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::uniform_size_t_distribution<std::mt19937::result_type> dist6(1,500);
    std::uniform_int_distribution<std::mt19937::result_type> dist6(1,1050);
    
    //unsigned size_t numberThreads = std::thread::hardware_concurrency();
    //std::cout << numberThreads << " concurrent threads are supported.\n";
    
    //For now we will assume a fixed number of threads (8).
    size_t numberThreads = 8;
    size_t chunk = 0;
    size_t extraChunk = 0;
    
    for (size_t i = 0; i < vector_size; ++i)
    {
        numericVector[i] = dist6(gen);
    }
    
    //for (auto i: numericVector)
        //std::cout << i << ", ";// <<std::endl;
    
    chunk =  vector_size/numberThreads;        
    if(vector_size%numberThreads != 0)
    {
        //This will be added to the size of the last vector/thread
        extraChunk = vector_size%numberThreads;       
    }

    std::vector<std::thread> threads;

    for(size_t i = 0; i < numberThreads; ++i)
    {
        //size_t extra = (i == (numberThreads - 1)) ? extraChunk : 0;
         
        threads.emplace_back(getMaxAndMin,std::cref(numericVector),chunk,extraChunk,i,std::ref(valuesMax),std::ref(valuesMin));
    }

    
    for (auto& t : threads)
    {
        t.join();
    }

    //*********************************************
    std::cout << "Using Threads: "<<std::endl;
    size_t max = 0; size_t min = 0;    
    getMax(valuesMax,0,valuesMax.size(),max);
    getMin(valuesMin,0,valuesMin.size(),min);      
    std::cout << "Max value: "<<max<<std::endl;
    std::cout << "Min value: "<<min<<std::endl;
    //*********************************************

    //*********************************************
    std::cout << "Using sequential: "<<std::endl;
    max = 0; min = 0;    
    getMax(numericVector,0,numericVector.size(),max);
    getMin(numericVector,0,numericVector.size(),min);    
    std::cout << "Max value: "<<max<<std::endl;
    std::cout << "Min value: "<<min<<std::endl;
    //*********************************************

    
    
    return 0;
}