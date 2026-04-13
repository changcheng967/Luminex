#pragma once

#include "types.h"

namespace luminex {

class Position;

// UCI loop - main entry point
void uci_loop();

// Process a single UCI command (for WASM use)
void process_uci_command(const std::string& line);

// UCI command handlers
void handle_uci();
void handle_isready();
void handle_ucinewgame();
void handle_position(Position& pos, const std::string& cmd);
void handle_go(Position& pos, const std::string& cmd);
void handle_setoption(const std::string& cmd);
void handle_stop();
void handle_quit();

// UCI output
void uci_send(const char* msg, ...);

// Thread-safe output for use by search thread
void uci_safe_output(const std::string& msg);

// Debug logging for search thread
void uci_debug_log(const char* format, ...);

} // namespace luminex
