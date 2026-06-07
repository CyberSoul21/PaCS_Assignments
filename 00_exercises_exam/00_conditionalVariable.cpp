#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <mutex>
#include <fstream>
#include <queue>
#include <condition_variable>
#include <chrono>


namespace
{
    std::mutex var_mutex;
    std::condition_variable cv;
    std::queue<int> sharedQueue;
    bool finished = false;
}


void producer()
{
    //generates number from 1 to 10
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(1,10);

    for (int i = 0; i < 10; ++i)
    {
        std::lock_guard<std::mutex> lck(var_mutex);
        //sharedQueue.push(dist6(gen));
        sharedQueue.push(i);
        cv.notify_one();
        std::cout<<"Produced: "<<i<<std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    finished = true;
    
}

void consumer()
{
    int data = 0;
    while (!finished)
    {
        std::unique_lock<std::mutex> lk(var_mutex);
        cv.wait(lk,[]{return !sharedQueue.empty();});
        data = sharedQueue.front();
        sharedQueue.pop();
        lk.unlock();
        std::cout<<"Consumed: "<<data<<std::endl;
    }
    
    
}



int main()
{
    int threads = 2;
    std::vector<std::thread> vectorThreads;
    vectorThreads.reserve(threads);

    vectorThreads.push_back(std::thread(producer));
    vectorThreads.push_back(std::thread(consumer));

    for(auto& thread:vectorThreads)
    {
        thread.join();
    }


    return 0;
}