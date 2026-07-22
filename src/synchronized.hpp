#pragma once

#include <mutex>
#include <utility>


namespace cllm
{

template <typename T>
class Synchronized
{
public:
    template <typename Ptr>
    class UniqueAccess
    {
    public:
        UniqueAccess(Ptr p, std::mutex& m): l_(m), p_(p) {}
        Ptr operator->() const noexcept { return p_; }
        auto& operator*() const noexcept { return *p_; }

    private:
        std::unique_lock<std::mutex> l_;
        Ptr p_;
    };

    using MutableAccess = UniqueAccess<T*>;
    using ConstAccess = UniqueAccess<const T*>;

public:
    template <typename... Args>
    explicit Synchronized(Args&&... args)
        : t_(std::forward<Args>(args)...) {}
    
    Synchronized(const Synchronized&) = delete;
    Synchronized& operator=(const Synchronized&) = delete;

    MutableAccess operator->() { return lock(); }
    ConstAccess operator->() const { return lock(); }

    [[nodiscard]] MutableAccess lock() { return {&t_, m_}; }
    [[nodiscard]] ConstAccess lock() const { return {&t_, m_}; }

    T& unsafe() noexcept { return t_; }
    const T& unsafe() const noexcept { return t_; }

private:
    mutable std::mutex m_;
    T t_;
};

}