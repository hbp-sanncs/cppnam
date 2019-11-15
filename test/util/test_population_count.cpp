/*
 *  CppNAM -- C++ Neural Associative Memory Simulator
 *  Copyright (C) 2016  Christoph Jenzen, Andreas Stöckel
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

#include "gtest/gtest.h"

#include <util/population_count.hpp>

namespace nam {

TEST(population_count, basic) {
	EXPECT_EQ(uint8_t(5), population_count<int8_t>(0x1F));
	EXPECT_EQ(uint8_t(1), population_count<int8_t>(0x1));
	EXPECT_EQ(uint64_t(1), population_count<uint64_t>(0x1));
	EXPECT_EQ(uint64_t(1), population_count<uint64_t>(0x1L << 63));
}

}
