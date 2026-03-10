#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <cstdarg>
#include <cstdio>

class Logger {
public:
    static Logger& instance();

    void init(int rank, const std::string& log_dir = "logs");

    void info (const char* fmt, ...);
    void warn (const char* fmt, ...);
    void error(const char* fmt, ...);

private:
    Logger() = default;
    ~Logger();

    void write(const char* level, const char* fmt, va_list ap);
    std::string timestamp();

    int          rank_  = 0;
    std::ofstream file_;
    std::mutex    mtx_;
};

#define LOG_INFO(...)  Logger::instance().info (__VA_ARGS__)
#define LOG_WARN(...)  Logger::instance().warn (__VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().error(__VA_ARGS__)
