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

#include <util/entropy.hpp>
#include <util/parameters.hpp>

namespace nam {

TEST(entropy, expected_false_positives)
{
	constexpr double EPS = 0.00001;
	const size_t N = 10, c = 2, d = 3, m = 10, n = 6;
	EXPECT_NEAR((n - d) * 0.424219774,
	            expected_false_positives(DataParameters(m, n, c, d, N)), EPS);
	EXPECT_NEAR(
	    (n - d) * 0.84039451,
	    expected_false_positives(
	        DataParameters().bits_out(n).ones_out(d).samples(N).canonicalize()),
	    EPS);
}
/*
TEST(entropy, entropy_hetero_uniform)
{

    int n = 3, m = 3, c = 1, d = 1, N = 3, false_pos = 2;
    EXPECT_EQ(,
              entropy_hetero_uniform(DataParameters(n, m, c, d, N), false_pos));
}*/
}
