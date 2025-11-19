/* Laboratory 4
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/
#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include<join_threads.hpp>
#include<threadsafe_queue.hpp>

//how to implemet std::promise and std::future inside of wait()

class thread_pool
{
  // Member declaration order matters!
  
  //Atomic flag that signals when the pool should stop atomic ensures thread-safe reads/writes without explicit locks
  std::atomic_bool done;
  std::atomic<unsigned> active_tasks;  // Track running tasks
  //Thread-safe queue holding tasks to execute
  //Each task is a std::function<void()> (callable with no parameters, returns void)
  threadsafe_queue<std::function<void()> > work_queue;
  std::vector<std::thread> threads;
  
  std::mutex completion_mutex;
  std::condition_variable completion_cv;

  join_threads joiner;

  
  using task_type = void();

  void worker_thread()
  {
    while(!done) // Keep running until pool stops
    {
      //task is a function object that holds whatever function/lambda that was submitted to the thread pool
      std::function<void()> task;
      if(work_queue.try_pop(task)) // Blocks until task available // Try to get a task (non-blocking)
      {

        task(); // Execute the task
        //active_tasks--;  // race condition between decrement and check:

        if(--active_tasks == 0)  //check //Decrement and check in ONE operation //
        {
          completion_cv.notify_all();  // Notify waiters
        } 
      }
      else // Queue is empty
      {
        // Give up CPU time slice to other threads
        std::this_thread::yield(); //TODO: Use condition variable instead of yield() for better efficiency
      }
    }

    // Process remaining tasks, it waits for the completion of all tasks in the queue
    std::function<void()> task;
    while(work_queue.try_pop(task))
    {
        task();
        //active_tasks--;
        
        if(--active_tasks == 0)//Decrement and check in ONE operation
        {
          completion_cv.notify_all();
        }
    }
  }

  public:

    thread_pool(size_t num_threads = std::thread::hardware_concurrency()):
      done(false), active_tasks(0), joiner(threads)
    {
      try
      {
        for(unsigned i=0;i<num_threads;++i)
        {
          threads.push_back(
                std::thread(&thread_pool::worker_thread,this));
        }
      }
      catch(...)
      {
        done=true; // Signal threads to stop
        throw; // Re-throw exception
      }
    }

    ~thread_pool()
    {
      wait();
      done=true;
    }


  void wait()
  {
      // wait for completion
      // active waiting
      //Wait for Queue to Empty, correct implementation
       std::unique_lock<std::mutex> lock(completion_mutex);
       completion_cv.wait(lock, [this] { 
          return active_tasks == 0 && work_queue.empty(); 
       });
      
      //Testing cycles with sleep.
     //while (active_tasks.load() != 0) {   //RACE CONDITION! (empty queue in not equal all tasks done)
     //    std::this_thread::sleep_for(std::chrono::milliseconds(10));
     // }
  }

    //Template accepts any callable (function, lambda, functor)
    template<typename F>
    void submit(F f)
    {
      ++active_tasks; 
      work_queue.push(std::function<void()>(f)); //Pushes to thread-safe queue and one waiting worker thread will pick it up
    }
};