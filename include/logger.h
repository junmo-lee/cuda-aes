#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <cstdarg>
#include <cstdio>

/**
 * @brief Thread-safe logging class with rank-aware output.
 * 
 * Provides a singleton logger to record information, warnings, and errors 
 * to both stdout and a log file specific to each MPI rank.
 */
class Logger {
public:
    /**
     * @brief Gets the singleton instance of the Logger.
     * @return Reference to the Logger instance.
     */
    static Logger& instance();

    /**
     * @brief Initializes the logger for a specific rank.
     * @param rank The MPI rank of the current process.
     * @param log_dir The directory where log files should be created.
     */
    void init(int rank, const std::string& log_dir = "logs");

    /**
     * @brief Logs an informational message.
     * @param fmt Printf-style format string.
     * @param ... Variable arguments for the format string.
     */
    void info (const char* fmt, ...);
    
    /**
     * @brief Logs a warning message.
     * @param fmt Printf-style format string.
     * @param ... Variable arguments for the format string.
     */
    void warn (const char* fmt, ...);
    
    /**
     * @brief Logs an error message.
     * @param fmt Printf-style format string.
     * @param ... Variable arguments for the format string.
     */
    void error(const char* fmt, ...);

private:
    Logger() = default;
    ~Logger();

    /**
     * @brief Writes a message to the log file and stdout with a prefix.
     * @param level Prefix indicating log level (e.g., INFO, WARN, ERROR).
     * @param fmt Printf-style format string.
     * @param ap Varargs list for formatting.
     */
    void write(const char* level, const char* fmt, va_list ap);
    
    /**
     * @brief Gets the current timestamp as a string.
     * @return Formatted timestamp string.
     */
    std::string timestamp();

    int          rank_  = 0; ///< MPI rank associated with this logger.
    std::ofstream file_;     ///< Log file stream.
    std::mutex    mtx_;      ///< Mutex for thread-safe access to the logger.
};

/**
 * @def LOG_INFO
 * @brief Macro for logging informational messages.
 */
#define LOG_INFO(...)  Logger::instance().info (__VA_ARGS__)

/**
 * @def LOG_WARN
 * @brief Macro for logging warning messages.
 */
#define LOG_WARN(...)  Logger::instance().warn (__VA_ARGS__)

/**
 * @def LOG_ERROR
 * @brief Macro for logging error messages.
 */
#define LOG_ERROR(...) Logger::instance().error(__VA_ARGS__)
