/* Laboratory 4
*  Wilson Javier Almario Rodriguez -> 962449
*  Álvaro Provencio Carralero -> 960625
*/
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template<typename T>
class threadsafe_queue
{
  private:
      // please complete
      mutable std::mutex m_mutex;
      std::queue<T> m_queue;
      std::condition_variable m_cond;

  public:
    threadsafe_queue() {}

    threadsafe_queue(const threadsafe_queue& other)
    {
	// please complete
        std::lock_guard<std::mutex> lk(other.m_mutex); //from slides
        //std::unique_lock<std::mutex> lock(other.m_mutex); //from book
        m_queue = other.m_queue;

    }

    threadsafe_queue& operator=(const threadsafe_queue&) = delete;

    void push(T new_value)
    {
	// please complete
        std::lock_guard<std::mutex> lk(m_mutex); //from slides
        //std::unique_lock<std::mutex> lock(m_mutex); //from book
        m_queue.push(new_value);
        m_cond.notify_one();
    }

    bool try_pop(T& value)
    {
	// please complete
        std::lock_guard<std::mutex> lk(m_mutex);
        if(m_queue.empty())
        {
            return false;
        }
        else
        {
            value = m_queue.front();
            m_queue.pop();
            return true;
        }     
    }

    void wait_and_pop(T& value)
    {
	// please complete
        std::unique_lock<std::mutex> lk(m_mutex);
        //avoid do operation on empty queue:
        m_queue.wait(lk,[this]{return !m_queue.empty();});
        value=m_queue.front();
        m_queue.pop();
    }

    std::shared_ptr<T> wait_and_pop()
    {
	// please complete
        std::shared_ptr<T> value;
        std::unique_lock<std::mutex> lk(m_mutex);
        //avoid do operation on empty queue:
        m_cond.wait(lk,[this]{return !m_queue.empty();});
        value = std::make_shared<T>(m_queue.front());
        m_queue.pop();
        return value;
    }

    bool empty() const
    {
	// please complete
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_queue.empty();
    }
};
