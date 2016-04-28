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

#include <cmath>

#include "entropy.hpp"
#include "ncr.hpp"

namespace nam {

double expected_false_positives(const DataParameters &params)
{
	size_t N = params.samples();
	double p = double(params.ones_in() * params.ones_out()) /
	           double(params.bits_in() * params.bits_out());
	return ((params.bits_out() - params.ones_out()) *
	        std::pow(1.0 - std::pow(1.0 - p, N), params.ones_in()));
}

double expected_entropy(const DataParameters &params)
{
	return entropy_hetero_uniform(params, expected_false_positives(params));
}

double entropy_hetero_uniform(const DataParameters &params,
                              double false_positives)
{
	double res = 0.0;
	for (size_t i = 0; i < params.ones_out(); i++) {
		res += std::log2(double(params.bits_out() - i) /
		                 double(params.ones_out() + false_positives - i));
	}
	return res * params.samples();
}

double entropy_hetero(const DataParameters &params,
                      const std::vector<SampleError> &errs)
{
	double ent = 0.0;
	for (auto &err : errs) {
		if (err.fn > 0) {
			ent +=
			    (lnncrr(params.bits_out(), params.ones_out()) -
			     lnncrr(err.fp + params.ones_out() - err.fn,
			            params.ones_out() - err.fn) -
			     lnncrr(params.bits_out() - err.fp - params.ones_out() + err.fn,
			            err.fn)) /
			    std::log(2.0);
		}
		else {
			for (size_t j = 0; j < params.ones_out(); j++) {
				ent += std::log2(double(params.bits_out() - j) /
				                 double(params.ones_out() + err.fp - j));
			}
		}
	}
	return ent;
}

double conventional_memory_entropy(const DataParameters &params)
{
	return params.bits_in() * lnncrr(params.bits_out(), params.ones_out()) /
	       std::log(2);
}
}

