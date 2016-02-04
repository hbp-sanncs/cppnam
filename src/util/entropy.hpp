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

#pragma once

#ifndef CPPNAM_UTIL_ENTROPY_HPP
#define CPPNAM_UTIL_ENTROPY_HPP

#include "parameters.hpp"

namespace nam {

/**
 * Calculates the to be expected average number of false positives for the
 * given data parameters.
 *
 * @param params structure containing the data parameters.
 * @return the expected number of false positives per sample.
 */
double expected_false_positives(const DataParameters &params);

/**
 * Calculates the expected entropy for data with the given parameters. See
 * expected_false_positives for a description.
 *
 * @param params structure containing the data parameters.
 */
double expected_entropy(const DataParameters &params);

/**
 * Calculates the entropy for an estimated number of false positives err (which
 * might be real-valued).
 *
 *   :param errs: estimated number of false positives per sample.
 *   :param n_samples: number of samples.
 *   :param n_bits_out: number of output bits.
 *   :param n_ones_out: number of ones in the output.
 */
double entropy_hetero_uniform(const DataParameters &params, double false_positives);

double entropy_hetero_uniform(const DataParameters &params)
{
	return entropy_hetero_uniform(params, expected_false_positives(params));
}

}

#endif /* CPPNAM_UTIL_ENTROPY_HPP */