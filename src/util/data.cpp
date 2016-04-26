/*
 *  CppNAM -- C++ Neural Associative Memory Simulator
 *  Copyright (C) 2016  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <map>
#include <limits>
#include <random>

#include "data.hpp"
#include "ncr.hpp"

namespace nam {

DataGenerator::DataGenerator(bool random, bool balance, bool unique)
    : m_seed(std::random_device()()),
      m_random(random),
      m_balance(balance),
      m_unique(unique)
{
}

DataGenerator::DataGenerator(size_t seed, bool random, bool balance,
                             bool unique)
    : m_seed(seed), m_random(random), m_balance(balance), m_unique(unique)
{
}
}
