#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>
#include <utility>


namespace cllm
{

template <typename T>
class Queue
{
public:
    Queue() = default;
    ~Queue() = default;

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    bool push(T value)
    {
        {
            std::lock_guard lock(mutex_);
            if (closed_) {
                return false;
            }
            queue_.push(std::move(value));
        }
        condition_.notify_one();
        return true;
    }

    std::optional<T> pop()
    {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [this] {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    void close()
    {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        condition_.notify_all();
    }

    bool closed()
    {
        std::lock_guard lock(mutex_);
        return closed_;
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<T> queue_;
    bool closed_ = false;
};

}
