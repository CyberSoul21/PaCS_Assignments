#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <mutex>
#include <fstream>
#include <algorithm>


namespace
{
    int vectorSize = 100;
//     std::mutex var_mutex;
//     std::condition_variable cv;
//     std::queue<int> sharedQueue;
//     bool finished = false;
}


template<typename T>
void insertion_sort(std::vector<T>& array)
{
    for(size_t i = 1; i < array.size(); ++i) 
    {
        for(size_t j = i; (j > 0) && (array[j-1] > array[j]); --j) 
        {
            std::swap(array[j], array[j-1]);
        }
    }
}


int main(int argc, char* argv[])
{

    //0. Creating data:

    std::vector<int> array(vectorSize);
    //std::vector<int> c(vectorSize);


    std::random_device rnd_device;
    std::mt19937 gen(rnd_device());
    std::uniform_int_distribution<> dis(0, 1000);

    for (int i = 0; i < vectorSize; ++i)
    {
        array[i] = dis(gen);
    }   



    //1. Computation of the ranges 
    //require to the user number of buckets:

    if (argc < 2)
    {
        std::cout << "Usage: ./main <number>\n";
        return 1;
    }

    int numberBuckets = std::atoi(argv[1]);

    std::cout << "Value received: " << numberBuckets << std::endl;

    //Maximum:

    auto it = std::max_element(array.begin(),array.end());

    int maxValue = 0;

    if ( it != array.end() )
    {
        maxValue = *it;
        std::cout<<"Maximum value stored: "<<maxValue<<std::endl;
    }

    int range = maxValue / numberBuckets;

    std::vector<std::vector<int>> buckets(numberBuckets);

    //First version, dprecated, because algorithm loops over the full array for every bucket.
    // for (int i = 0; i < numberBuckets; i++)
    // {

    //     int lower = i * range;
    //     int upper = (i == numberBuckets - 1)
    //                     ? maxValue
    //                     : lower + range - 1;

    //     for (int j = 0; j < array.size(); j++)
    //     {
    //         if (array[j] >= lower && array[j] <= upper)
    //         {
    //             buckets[i].push_back(array[j]);
    //         }
            
    //     }
        
    // }

    //better version
    for (int value : array)
    {
        int bucketIndex = value / range;

        if (bucketIndex >= numberBuckets)
        {
            bucketIndex = numberBuckets - 1;
        }

        buckets[bucketIndex].push_back(value);
    }
    
    //create threads
    std::vector<std::thread> threads;

    for (size_t i = 0; i < numberBuckets; i++)
    {
        threads.emplace_back(insertion_sort<int>,std::ref(buckets[i]));
    }
    
    for (auto& t : threads)
    {
        t.join();
    }

    // Concatenate all sorted buckets
    std::vector<int> sortedArray;

    for (const auto& bucket : buckets)
    {
        sortedArray.insert(sortedArray.end(), bucket.begin(), bucket.end());
    }


    std::cout << "Final sorted array:\n";

    for (int value : sortedArray)
    {
        std::cout << value << " ";
    }

    std::cout << std::endl;    


    //b. 
    //-when number of buckets > size of vector
    //-if all number are inside in one bucket...A pathological case is when the data is highly unbalanced between buckets.

    return 0;
}