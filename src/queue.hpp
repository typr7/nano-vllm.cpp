#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <utility>


namespace cllm
{

template <typename T>
class Queue
{
public:
    Queue() = default;

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    void push(T value)
    {
        {
            const std::lock_guard lock(mutex_);
            if (closed_) {
                throw std::logic_error("Cannot push to a closed queue");
            }
            queue_.push_back(std::move(value));
        }
        condition_.notify_one();
    }

    void push_front(T value)
    {
        {
            const std::lock_guard lock(mutex_);
            if (closed_) {
                throw std::logic_error("Cannot push to a closed queue");
            }
            queue_.push_front(std::move(value));
        }
        condition_.notify_one();
    }

    std::optional<T> pop(std::stop_token stop_token = {})
    {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, stop_token, [this] {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop_front();
        return value;
    }

    void close() noexcept
    {
        {
            const std::lock_guard lock(mutex_);
            closed_ = true;
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable_any condition_;
    std::deque<T> queue_;
    bool closed_{false};
};

}
