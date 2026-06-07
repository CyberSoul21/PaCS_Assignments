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

bool is_prime(size_t n)
{
    if (n <= 3)
    {
        return n > 1;
    }
    if (( n % 2 == 0 ) || (( n % 3 )==0))
    {
        return false;
    }

    for(size_t i = 5; i*i <= n; i+=6)
    {
        if (( n % i == 0 ) || (( n % (i +2) )==0))
        {
            return false;
        }
    }

    return true;
}

void getPrimeAndLargerThanGivenValue(const size_t id_thread, 
                                     const size_t Chunk, 
                                     const size_t extraChunk, 
                                     const size_t givenValue, 
                                     const std::vector<size_t>& array, 
                                     std::vector<size_t>& result)
{
    size_t begin = id_thread*Chunk;
    size_t size  = Chunk + (id_thread < extraChunk ? 1 : 0);
    size_t end   = begin + size;

    for(size_t i = begin; i < end; ++i)
    {
        if ((is_prime(array[i])) && (array[i] > givenValue))
        {
            std::lock_guard<std::mutex> lck(var_mutex);
            {
                result.push_back(array[i]);
            }
        }
    }

}


int main()
{
    size_t vector_size = 200000000;
    size_t givenValue = 0;
    std::vector<size_t> numericVector(vector_size);
    std::vector<size_t> resultArray;

    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(1,1050);
    
    //For now we will assume a fixed number of threads (8).
    size_t numberThreads = 16;
    size_t chunk = 0;
    size_t extraChunk = 0;
    
    //TODO: Parallelize
    for (size_t i = 0; i < vector_size; ++i)
    {
        numericVector[i] = dist6(gen);
    }
    
    //for (auto i: numericVector)
    //    std::cout << i << ", ";// <<std::endl;
    
    chunk =  vector_size/numberThreads;        
    if(vector_size%numberThreads != 0)
    {
        //This will be added to the size of the last vector/thread
        extraChunk = vector_size%numberThreads;       
    }

    std::vector<std::thread> threads;    


    std::cout<<"Given Value: ";
    std::cin>>givenValue;
    std::cout<<std::endl;

    //******************************************************************************************************* */
    //Parallel

    for (size_t i = 0; i < numberThreads; ++i)
    {
        threads.emplace_back(getPrimeAndLargerThanGivenValue,i,chunk,extraChunk,givenValue,std::cref(numericVector),std::ref(resultArray)); 
    }

    for (auto& t : threads)
    {
        t.join();
    }
    std::cout<<"**************¡RESULT!**************"<<std::endl;
    for (auto i: resultArray)
        std::cout << i << ", ";// <<std::endl;
    std::cout<<std::endl;

    //*******************************************************************************************************


    //******************************************************************************************************* */
    //Sequential
    //getPrimeAndLargerThanGivenValue(0,numericVector.size(),0,givenValue,numericVector,std::ref(resultArray));

    // std::cout<<"**************¡RESULT!**************"<<std::endl;
    // for (auto i: resultArray)
    //     std::cout << i << ", ";// <<std::endl;
    // std::cout<<std::endl;
    //******************************************************************************************************* */
    
    



    return 0;
}