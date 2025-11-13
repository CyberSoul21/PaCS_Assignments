#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include<join_threads.hpp>
#include<threadsafe_queue.hpp>

//how to implemet std::promise and std::future inside of wait()

class thread_pool
{
  //Atomic flag that signals when the pool should stop atomic ensures thread-safe reads/writes without explicit locks
  std::atomic_bool done;
  //Thread-safe queue holding tasks to execute
  //Each task is a std::function<void()> (callable with no parameters, returns void)
  threadsafe_queue<std::function<void()> > work_queue;
  std::vector<std::thread> threads;
  join_threads joiner;


  std::atomic<unsigned> active_tasks;  // Track running tasks
  std::mutex completion_mutex;
  std::condition_variable completion_cv;

  
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
        active_tasks--;

        if(active_tasks == 0)  //check
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
        active_tasks--;
        
        if(active_tasks == 0)
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
      /*
      while (!work_queue.empty()) {   //RACE CONDITION! (empty queue in not equal all tasks done)
          std::this_thread::yield();
      }*/
      
      //Wait for Queue to Empty
      std::unique_lock<std::mutex> lock(completion_mutex);
      completion_cv.wait(lock, [this] { 
          return active_tasks == 0 && work_queue.empty(); 
      });
  }

    //Template accepts any callable (function, lambda, functor)
    template<typename F>
    void submit(F f)
    {
      work_queue.push(std::function<void()>(f)); //Pushes to thread-safe queue and one waiting worker thread will pick it up
    }
};