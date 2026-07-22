#pragma once

#include <source_location>
#include <string_view>


namespace cllm
{

enum class LogLevel
{
    INFO,
    ERROR,
    DEBUG,
};

class Logger
{
public:

    static void log(
        LogLevel level,
        std::string_view message,
        const std::source_location& location = std::source_location::current()
    ) noexcept;

    static void info(
        std::string_view message,
        const std::source_location& location = std::source_location::current()
    ) noexcept
    {
        log(LogLevel::INFO, message, location);
    }

    static void error(
        std::string_view message,
        const std::source_location& location = std::source_location::current()
    ) noexcept
    {
        log(LogLevel::ERROR, message, location);
    }

    static void debug(
        std::string_view message,
        const std::source_location& location = std::source_location::current()
    ) noexcept
    {
        log(LogLevel::DEBUG, message, location);
    }
};

}
