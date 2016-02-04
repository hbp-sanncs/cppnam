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

#ifndef CPPNAM_UTIL_COMPARISON_HPP
#define CPPNAM_UTIL_COMPARISON_HPP

#include <cstdint>

namespace nam {
/**
 * The Comperator class can be used to assemble comperator operators, such as
 * equals, smaller or larger than.
 *
 * @tparam T is the type for which the comperator is being instantiated.
 */
template <typename T>
struct Comperator {
	/**
	 * Internally used functor.
	 */
	template <int Threshold, typename F>
	struct ComperatorFunctor {
		const F &f;
		const int res;

		ComperatorFunctor(const F &f, int res = 0) : f(f), res(res) {}
		ComperatorFunctor<Threshold, F> operator()(const T &t1,
		                                           const T &t2) const
		{
			if (res == 0) {
				return make_comperator<Threshold, F>(f, (f(t1, t2)));
			}
			return make_comperator<Threshold, F>(f, res);
		}

		bool operator()() const { return res >= Threshold; }
	};

	template <int Threshold, typename F>
	static ComperatorFunctor<Threshold, F> make_comperator(const F &f,
	                                                       int res = 0)
	{
		return ComperatorFunctor<Threshold, F>(f, res);
	}

	static auto smaller(const T &t1, const T &t2)
	{
		return make_comperator<1>([](const T &t1, const T &t2) {
			return (t1 < t2) ? 1 : ((t1 == t2) ? 0 : -1);
		})(t1, t2);
	}

	static auto smaller_equals(const T &t1, const T &t2)
	{
		return make_comperator<0>([](const T &t1, const T &t2) {
			return (t1 < t2) ? 1 : ((t1 == t2) ? 0 : -1);
		})(t1, t2);
	}

	static auto larger(const T &t1, const T &t2)
	{
		return make_comperator<1>([](const T &t1, const T &t2) {
			return (t1 > t2) ? 1 : ((t1 == t2) ? 0 : -1);
		})(t1, t2);
	}

	static auto larger_equals(const T &t1, const T &t2)
	{
		return make_comperator<0>([](const T &t1, const T &t2) {
			return (t1 > t2) ? 1 : ((t1 == t2) ? 0 : -1);
		})(t1, t2);
	}

	static auto equals(const T &t1, const T &t2)
	{
		return make_comperator<0>([](const T &t1, const T &t2) {
			return (t1 == t2) ? 0 : -1;
		})(t1, t2);
	}
};
}

#endif /* CPPNAM_UTIL_COMPARISON_HPP */