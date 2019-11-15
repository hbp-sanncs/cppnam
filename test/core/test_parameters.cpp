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

#include <cstdint>
#include <limits>

#include <core/entropy.hpp>
#include <core/parameters.hpp>

namespace nam {

TEST(parameters, optimal_sample_count)
{
	EXPECT_EQ(52u, DataParameters::optimal_sample_count(
	                  DataParameters(16, 16, 2, 2, 0)));
	EXPECT_EQ(62u, DataParameters::optimal_sample_count(
	                  DataParameters(32, 32, 4, 4, 0)));
}

TEST(parameters, optimal)
{
	EXPECT_EQ(52u, DataParameters::optimal(16).samples());
	EXPECT_EQ(97u, DataParameters::optimal(32).samples());
	EXPECT_EQ(2u, DataParameters::optimal(16).ones_out());
	EXPECT_EQ(2u, DataParameters::optimal(32).ones_out());
}
}
