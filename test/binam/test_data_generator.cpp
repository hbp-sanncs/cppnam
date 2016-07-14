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

#include <gtest/gtest.h>

#include <algorithm>

#include <cppnam/binam/ncr.hpp>
#include <cppnam/binam/data_generator.hpp>

namespace nam {

static void check_row_sum(size_t n_bits, size_t n_ones, size_t n_samples)
{
	auto data = DataGenerator().seed(42198).generate(n_bits, n_ones, n_samples);
	EXPECT_EQ(n_samples, data.rows());
	EXPECT_EQ(n_bits, data.cols());
	for (size_t i = 0; i < data.rows(); i++) {
		size_t sum = 0;
		for (auto it = data.begin_row(i); it < data.end_row(i); it++) {
			sum += *it ? 1 : 0;
		}
		EXPECT_EQ(n_ones, sum);
	}
}

static void check_col_sum(size_t n_bits, size_t n_ones, size_t n_samples)
{
	auto data = DataGenerator().seed(42198).generate(n_bits, n_ones, n_samples);
	EXPECT_EQ(n_samples, data.rows());
	EXPECT_EQ(n_bits, data.cols());
	std::vector<size_t> sums(n_bits, 0);
	for (size_t i = 0; i < data.rows(); i++) {
		for (size_t j = 0; j < data.cols(); j++) {
			sums[j] += data(i, j) ? 1 : 0;
		}
		size_t min = n_samples;
		size_t max = 0;
		for (size_t j = 0; j < data.cols(); j++) {
			min = std::min(min, sums[j]);
			max = std::max(max, sums[j]);
		}
		EXPECT_GE(3, max - min);  // Allow a little sloppyness while balancing
		                          // -- the greedy approach used in the
		                          // algorithm is not perfect
	}
}

static void check_permutations(size_t n_bits, size_t n_ones, size_t mul = 1)
{
	const size_t n_samples = ncr(n_bits, n_ones) * mul;
	auto data = DataGenerator().seed(42198).generate(n_bits, n_ones, n_samples);
	std::vector<std::vector<bool>> res(n_samples);
	for (size_t i = 0; i < n_samples; i++) {
		res[i] = std::vector<bool>(n_bits);
		for (size_t j = 0; j < n_bits; j++) {
			res[i][j] = data(i, j);
		}
	}
	std::sort(res.begin(), res.end());

	auto it = res.begin();
	while (it != res.end()) {
		auto p = std::equal_range(it, res.end(), *it);
		it = p.second;

		EXPECT_EQ(mul, p.second - p.first);
	}
}

TEST(DataGenerator, row_sum)
{
	for (size_t i = 0; i <= 10; i++) {
		check_row_sum(10, i, 100);
	}
	check_row_sum(60, 20, 1000);
	check_row_sum(384, 60, 1000);
}

TEST(DataGenerator, col_sum)
{
	for (size_t i = 0; i <= 10; i++) {
		check_row_sum(10, i, 100);
	}
	check_col_sum(60, 20, 1000);
	check_col_sum(384, 60, 1000);
}

TEST(DataGenerator, unique)
{
	for (size_t mul = 1; mul <= 10; mul++) {
		for (size_t i = 0; i <= 10; i++) {
			check_permutations(10, i, 1);
		}
	}
	check_permutations(100, 3, 2);
}
}
