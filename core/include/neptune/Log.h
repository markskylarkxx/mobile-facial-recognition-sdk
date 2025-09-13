#pragma once
#include <string>
#include <iostream>

namespace neptune {

/**
 * @class Log
 * @brief A simple utility class for logging messages with different severities.
 *
 * This class provides static methods to print messages to the console
 * or to platform-specific loggers. The implementation will be extended
 * in the platform-specific wrappers to route messages to tools like
 * Android's Logcat or iOS's unified logging system.
 */
class Log {
public:
    /**
     * @brief Logs an informational message.
     * @param tag The tag to identify the source of the message.
     * @param message The message content.
     */
    static void info(const std::string& tag, const std::string& message);

    /**
     * @brief Logs a warning message.
     * @param tag The tag to identify the source of the message.
     * @param message The message content.
     */
    static void warn(const std::string& tag, const std::string& message);

    /**
     * @brief Logs an error message.
     * @param tag The tag to identify the source of the message.
     * @param message The message content.
     */
    static void error(const std::string& tag, const std::string& message);

    /**
     * @brief Logs a debug message.
     * @param tag The tag to identify the source of the message.
     * @param message The message content.
     */
    static void debug(const std::string& tag, const std::string& message);
};

} // namespace neptune