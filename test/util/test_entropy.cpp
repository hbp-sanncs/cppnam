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

TEST(entropy, entropy_hetero)
{
	constexpr double EPS = 0.00001;
	DataParameters params(16, 16, 3, 3, 3);
	std::vector<SampleError> errs = {SampleError(1, 0), SampleError(0, 0),
	                                 SampleError(2, 0)};
	std::vector<SampleError> errs2 = {SampleError(1), SampleError(0),
	                                  SampleError(2)};
	EXPECT_NEAR(entropy_hetero(params, errs), 22.06592095594754, EPS);
	EXPECT_NEAR(entropy_hetero(params, errs2), 22.06592095594754, EPS);
}

TEST(entropy, entropy_hetero_uniform)
{
	constexpr double EPS = 0.00001;
	DataParameters params(16, 16, 3, 3, 3);
	std::vector<SampleError> err = {SampleError(1.5), SampleError(1.5),
	                                SampleError(1.5)};
	EXPECT_NEAR(entropy_hetero(params, err),
	            entropy_hetero_uniform(params, 1.5), EPS);
}
}
