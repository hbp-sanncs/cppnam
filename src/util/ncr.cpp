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

#include <cmath>
#include <limits>

#include "ncr.hpp"

namespace nam {



uint64_t ncr(int n, int r)
{
	// Special case handling
	if ((n < 0) || (n < r)) {
		return 0;
	}

	// Take the shorter path
	if (r > n / 2) {
		return ncr(n, n - r);
	}

	// Actual computation
	uint64_t res = 1;
	for (int i = 1; i <= r; i++) {
		res = (res * (n + 1 - i)) / i;
	}

	return res;
}

template <typename IntType>
static IntType ncr_clamped(int n, int r)
{
	static constexpr IntType max = std::numeric_limits<IntType>::max();
	static constexpr double log_max = std::log(max);

	// Special case handling
	if ((n < 0) || (n < r) || (r < 0)) {
		return 0;
	}
	if ((n == r) || (r == 0)) {
		return 1;
	}

	// Use lgamma provided by lnnccr to calculate the result
	double res = lnncrr(n, r);
	if (res > log_max) {
		return max;
	}

	// Apply the exponential
	res = std::round(std::exp(res));
	if (res > max) {
		return max;
	}
	return res;
}

uint32_t ncr_clamped32(int n, int r) { return ncr_clamped<uint32_t>(n, r); }

uint64_t ncr_clamped64(int n, int r) { return ncr_clamped<uint64_t>(n, r); }

double lnncrr(double x, double y)
{
	return std::lgamma(x + 1.0) - std::lgamma(y + 1.0) -
	       std::lgamma(x - y + 1.0);
}
}
