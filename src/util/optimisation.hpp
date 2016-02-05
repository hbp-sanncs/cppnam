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

#ifndef CPPNAM_UTIL_OPTIMISATION
#define CPPNAM_UTIL_OPTIMISATION

#include <cstdint>
#include <cmath>

namespace nam {

/**
* Implementation of Golden section search
* https://en.wikipedia.org/wiki/Golden_section_search
*/
template <typename Function>
static double find_minimum_unimodal(Function f, double a, double b,
                                    double tolerance = 1.0)
{
	static constexpr double gr = 0.5 * (std::sqrt(5) - 1);
	double c = b - gr * (b - a);
	double d = a + gr * (b - a);
	while (std::abs(c - d) > tolerance) {
		const double fc = f(c);
		const double fd = f(d);
		if (fc < fd) {
			b = d;
			d = c;
			c = b - gr * (b - a);
		}
		else {
			a = c;
			c = d;
			d = a + gr * (b - a);
		}
	}
	return (b + a) / 2.0;
}
}

#endif /* CPPNAM_UTIL_OPTIMISATION */