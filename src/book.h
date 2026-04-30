#pragma once

#include "types.h"

namespace luminex {

class Position;

void init_book();
Move book_probe(const Position& pos);
void book_new_game();

} // namespace luminex
