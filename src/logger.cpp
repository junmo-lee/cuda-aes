#include "logger.h"

#include <chrono>
#include <ctime>
#include <sys/stat.h>

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

void Logger::init(int rank, const std::string& log_dir) {
    rank_ = rank;
    mkdir(log_dir.c_str(), 0755);

    std::string path = log_dir + "/node_" + std::to_string(rank) + ".log";
    file_.open(path, std::ios::app);
    info("Logger initialised (rank=%d)", rank);
}

Logger::~Logger() {
    if (file_.is_open()) file_.close();
}

std::string Logger::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));
    return buf;
}

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

void Logger::info(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); write("INFO",  fmt, ap); va_end(ap);
}
void Logger::warn(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); write("WARN",  fmt, ap); va_end(ap);
}
void Logger::error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); write("ERROR", fmt, ap); va_end(ap);
}
