#include "logger.h"

#include <chrono>
#include <ctime>
#include <sys/stat.h>

/**
 * @brief Returns the singleton instance of the Logger.
 * 
 * @return Reference to the Logger instance.
 */
Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

/**
 * @brief Initialises the logger for a specific MPI rank.
 * 
 * Creates the log directory if it doesn't exist and opens the log file
 * for this node.
 * 
 * @param rank The MPI rank of the current process.
 * @param log_dir The directory where log files should be stored.
 */
void Logger::init(int rank, const std::string& log_dir) {
    rank_ = rank;
    mkdir(log_dir.c_str(), 0755);

    std::string path = log_dir + "/node_" + std::to_string(rank) + ".log";
    file_.open(path, std::ios::app);
    info("Logger initialised (rank=%d)", rank);
}

/**
 * @brief Logger destructor.
 * 
 * Closes the log file if it is open.
 */
Logger::~Logger() {
    if (file_.is_open()) file_.close();
}

/**
 * @brief Generates a formatted timestamp string.
 * 
 * @return A string containing the current date and time.
 */
std::string Logger::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));
    return buf;
}

/**
 * @brief Core logging function that writes a message to both the log file and stdout.
 * 
 * Includes timestamp, log level, and MPI rank in the output.
 * 
 * @param level The log level string (e.g., "INFO", "WARN", "ERROR").
 * @param fmt Format string for the message.
 * @param ap Variable argument list.
 */
void Logger::write(const char* level, const char* fmt, va_list ap) {
    char msg[2048];
    vsnprintf(msg, sizeof(msg), fmt, ap);

    std::string line = "[" + timestamp() + "][" + level + "][rank="
                     + std::to_string(rank_) + "] " + msg + "\n";

    std::lock_guard<std::mutex> lk(mtx_);
    if (file_.is_open()) { file_ << line; file_.flush(); }
    fputs(line.c_str(), stdout);
    fflush(stdout);
}

/**
 * @brief Logs an informational message.
 * 
 * @param fmt Format string for the message.
 */
void Logger::info(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); write("INFO",  fmt, ap); va_end(ap);
}

/**
 * @brief Logs a warning message.
 * 
 * @param fmt Format string for the message.
 */
void Logger::warn(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); write("WARN",  fmt, ap); va_end(ap);
}

/**
 * @brief Logs an error message.
 * 
 * @param fmt Format string for the message.
 */
void Logger::error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); write("ERROR", fmt, ap); va_end(ap);
}
