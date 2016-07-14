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

/**
 * @file ncr.hpp
 *
 * Implementations of the binomial coefficient used in the data generator and
 * for the entropy calculations.
 *
 * @author Andreas Stöckel
 */

#include <cstdint>

namespace nam {
/**
 * Calculates the binomial coefficient, the number of possibilities to draw r
 * elements from a set of n.
 *
 * @param n is the size of the set from which elements are drawn.
 * @param r is the number of elements that are being drawn.
 */
uint64_t ncr(int n, int r);

/**
 * Calculates the number of possibilities to draw r elements from a set of n,
 * clamps the result to the range of a 32-bit unsigned integer. Internally uses
 * lnncrr.
 *
 * @param n is the size of the set from which elements are drawn.
 * @param r is the number of elements that are being drawn.
 */
uint32_t ncr_clamped32(int n, int r);

/**
 * Calculates the number of possibilities to draw r elements from a set of n,
 * clamps the result to the range of a 64-bit unsigned integer. Internally uses
 * lnncrr.
 *
 * @param n is the size of the set from which elements are drawn.
 * @param r is the number of elements that are being drawn.
 */
uint64_t ncr_clamped64(int n, int r);

/**
 * Real-valued generalisation of the binomial-coefficient. Returns the natural
 * logarithm of the result.
 */
double lnncrr(double x, double y);
}
