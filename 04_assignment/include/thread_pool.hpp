#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include<join_threads.hpp>
#include<threadsafe_queue.hpp>

class thread_pool
{
//added
  std::atomic<bool> _done;
  size_t _thread_count;
  threadsafe_queue<std::function<void()>> _work_queue;
  std::vector<std::thread> _threads;
//end of added
  join_threads _joiner;

  using task_type = void();

  void worker_thread()
  {
    while(!_done)
    {
      std::function<task_type> task;
      if(_work_queue.try_pop(task)){
        task();
      } else {
        std::this_thread::yield();
      }
    }
  }

  public:
  thread_pool(size_t num_threads = std::thread::hardware_concurrency())
    : // please complete
    _done(false), _thread_count(num_threads), _joiner(_threads)
  {
      // complete
    for(size_t i = 0; i < _thread_count; ++i){
      _threads.push_back(std::thread(&thread_pool::worker_thread,this));
    }
  }

  ~thread_pool()
  {
    wait();
  }

  void wait()
  {
      // wait for completion
      // active waiting
      while (!_work_queue.empty()) {
          std::this_thread::yield();
      }

      _done = true; 
  }

  template<typename F>
    void submit(F f)
    {
	// please complete
      _work_queue.push(std::function<task_type>(f));
    }
};
