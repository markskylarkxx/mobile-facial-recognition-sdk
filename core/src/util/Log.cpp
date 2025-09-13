#include "neptune/Log.h"

namespace neptune {

void Log::info(const std::string& tag, const std::string& message) {
    std::cout << "[INFO][" << tag << "] " << message << std::endl;
}

void Log::warn(const std::string& tag, const std::string& message) {
    std::cout << "[WARN][" << tag << "] " << message << std::endl;
}

void Log::error(const std::string& tag, const std::string& message) {
    std::cerr << "[ERROR][" << tag << "] " << message << std::endl;
}

void Log::debug(const std::string& tag, const std::string& message) {
    std::cout << "[DEBUG][" << tag << "] " << message << std::endl;
}

} // namespace neptune