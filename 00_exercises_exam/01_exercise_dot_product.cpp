#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <fstream>


/*
*Exercise 1:
*The dot product algorithm takes two vectors of the same length and returns a single number. The number
*is the sum of the products of the corresponding entries in the input vectors.
*In C++, the algorithm can be coded as follows:
*
    a. Implement the doc product using threads and static partitioning.
    b. Implement the doc product assuming you have the thread pool and the thread-safe queue from
    Laboratory 4.
    c. For the thread-pool version, would all tasks perform the same ammount of work?
*
*/


namespace
{
    int vectorSize = 100000;
    int result = 0;
    std::mutex var_mutex;
}

template<typename T>
void dot_product(const std::vector<T>& a, const std::vector<T>& b, const T begin, const T end, T& result)
{
    if(a.size() != b.size())
    {
        //std::cout<<"Error..."<<std::endl;
        throw std::runtime_error("Vectors must be same size");
    }
    
    //T dot_p{};

    for (size_t i = begin; i < end; i++)
    {
        //dot_p += (a[i]*b[i]);
        //m.lock();
        std::lock_guard<std::mutex> lk(var_mutex); //RAII: auto-unlock
        result += (a[i]*b[i]); 
        //m.unlock();
    }
    
    //return dot_p;
}


int main()
{
    //Testing template
    std::cout<<"Testing Template"<<std::endl;

    std::vector<int> a(vectorSize);
    std::vector<int> b(vectorSize);
    //std::vector<int> c(vectorSize);


    std::random_device rnd_device;
    std::mt19937 gen(rnd_device());
    std::uniform_int_distribution<> dis(0, 100);

    for (int i = 0; i < vectorSize; ++i)
    {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }   

    //sequential
    //dot_product(a,b,0,vectorSize,std::ref(result));

    //a. Implemented
    std::thread first_half(dot_product<int>,std::ref(a),std::ref(b),0,vectorSize/2,std::ref(result));
    std::thread second_half(dot_product<int>,std::ref(a),std::ref(b),vectorSize/2,vectorSize,std::ref(result));
    first_half.join();
    second_half.join();


    //b. Thread pool




    std::cout<<"Result Threads: "<<result<<std::endl;

    return 0;

    //Possible issues using mutexes:
    //1. Forgetting unlock, no release causing a deadlock
    //2. Throwing exeption, unexpected termination
    //3. Nested funtions 
}


// class mutexExample
// {

//     public:

//         //1. Avoid deadlocks with std::lock_guard<std:mutex> lock(mutex)
//         void WriteToFile(const std::string& message)
//         {
//             //Mutex is used to protec access to file which is shared acrosss threads
//             static std::mutex mutex;

//             //Lock |mutex| before accessing |file|.
//             std::lock_guard<std::mutex> lock(mutex);

//             //Try to open the file("example.txt");
//             std::ofstream file("example.txt");
//             if(!file.is_open())
//             {
//                 //Mutex released here:
//                 throw std::runtime_error("Unable to open the fiel");
//             }

//             // Write |message| to |file|
//             file << message << std::endl;

//             //|file| will be closed first when leaving scope (regardless of exeption)
//             //mutex will be unlocked second (from lock destructor) when leaving scope
//             // (regardless of exception).
//         }

//         //2. std:recursive_mutex, allow to aquire an already acquired mutex by the same thread
//         std::recursive_mutex myMutexRec;
//         int sharedVariable;

//         void doSomething1() {/*doing something*/};
//         void doSomething2() {/*doing something*/};

//         void func2()
//         {
//             std::lock_guard<std::recursive_mutex> lock(myMutexRec);
//             sharedVariable++;
//             doSomething2();
//         }

//         void func1()
//         {
//             std::lock_guard<std::recursive_mutex> lock(myMutexRec);
//             doSomething2();
//             sharedVariable++;
//             func2();
//         }
// };

// class otherDeadlock
// {
//     private:
//         std::mutex mutex1;
//         std::mutex mutex2;

//         void doSomething1() {/*doing something*/};
//         void doSomething2() {/*doing something*/};        

//         void func2() 
//         {
//             std::lock_guard<std::mutex> lck1(mutex1); //lock mutex1 first
//             std::lock_guard<std::mutex> lck1(mutex2); //then mutex2
//             doSomething2();
//         }

//         void func1() 
//         {
//             std::lock_guard<std::mutex> lck1(mutex2); //locks mutex2 first 
//             std::lock_guard<std::mutex> lck1(mutex1); //then mutex 1
//             doSomething1();            
            
//         }        

//     public:

//         int otherDeadlock_main()
//         {
//             //thread 1 aquires mutex1 -> waiting for mutex2
//             std::thread t1(func1);
//             //thread 2 aquires mutex2 -> waiting for mutex1
//             std::thread t2(func2);

//             //both waiting for each other  -> DEADLOCK forever 

//             //SOLUTION: use different name
//             //std::lock_guard<std::mutex> lck1(mutex2); //locks mutex2 first 
//             //std::lock_guard<std::mutex> lck2(mutex1); //then mutex 1
//             return 0;
//         }
// };

