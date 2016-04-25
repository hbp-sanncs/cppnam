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

#ifndef CPPNAM_POPULATION_COUNT_HPP
#define CPPNAM_POPULATION_COUNT_HPP

#include <stddef.h>

#include <cstdint>

namespace nam {

template <typename T>
size_t population_count(T i)
{
	size_t res;
	for (size_t j = 0; j < sizeof(T) * 8; j++) {
		res += (i & (1 << j)) ? 1 : 0;
	}
	return res;
}

template <>
inline size_t population_count<int8_t>(int8_t i)
{
	return __builtin_popcount(i);
}

template <>
inline size_t population_count<uint8_t>(uint8_t i)
{
	return __builtin_popcount(i);
}

template <>
inline size_t population_count<int16_t>(int16_t i)
{
	return __builtin_popcount(i);
}

template <>
inline size_t population_count<uint16_t>(uint16_t i)
{
	return __builtin_popcount(i);
}

template <>
inline size_t population_count<int32_t>(int32_t i)
{
	return __builtin_popcountl(i);
}

template <>
inline size_t population_count<uint32_t>(uint32_t i)
{
	return __builtin_popcountl(i);
}

template <>
inline size_t population_count<int64_t>(int64_t i)
{
	return __builtin_popcountll(i);
}

template <>
inline size_t population_count<uint64_t>(uint64_t i)
{
	return __builtin_popcountll(i);
}

}

#endif /* CPPNAM_POPULATION_COUNT_HPP */
