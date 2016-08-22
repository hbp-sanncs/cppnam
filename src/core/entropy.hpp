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

#ifndef CPPNAM_CORE_ENTROPY_HPP
#define CPPNAM_CORE_ENTROPY_HPP

#include <vector>

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
 * @param params structure containing the data parameters.
 * @param errs estimated number of false positives per sample.
 */
double entropy_hetero_uniform(const DataParameters &params,
                              double false_positives);

/**
 * Datastructure containing the number of false positives and false negatives
 * per sample.
 */
struct SampleError {
	double fp, fn;

	SampleError(double fp = 0.0, double fn = 0.0) : fp(fp), fn(fn) {}
};

struct ExpResults {
	double Info, fp, fn;

	ExpResults(double Info, double fp, double fn)
	    : Info(Info), fp(fp), fn(fn){};
	ExpResults(double Info, SampleError se)
	    : Info(Info), fp(se.fp), fn(se.fn){};
	ExpResults(){};
	void print()
	{
		std::cout << "Info :" << Info << " pos:" << fp << " neg: " << fn
		          << std::endl;
	}
};

/**
 * Calculates the entropy from an errors-per sample matrix (returned by
 * analyseSampleErrors) and for the given output vector size and the mean
 * number of set bits. All values may also be real/floating point numbers,
 * a corresponding real version of the underlying binomial coefficient is
 * used.
 *
 * @param errs errs is a vector of SampleError containing "fn" and
 * "fp" entries, where "fn" corresponds to the number of false negatives and
 * "fp" to the number of false positives.
 */
double entropy_hetero(const DataParameters &params,
                      const std::vector<SampleError> &errs);

/**
 * Calculates storage capacity of a conventional MxN ROM holding data with
 * the specification
 */
double conventional_memory_entropy(const DataParameters &params);
}

#endif /* CPPNAM_CORE_ENTROPY_HPP */