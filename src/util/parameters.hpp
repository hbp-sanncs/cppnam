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

#ifndef CPPNAM_UTIL_PARAMETERS_HPP
#define CPPNAM_UTIL_PARAMETERS_HPP

#include <cstddef>

namespace nam {

class DataParameters {
private:
	size_t m_bits_in;
	size_t m_bits_out;
	size_t m_ones_in;
	size_t m_ones_out;
	size_t m_samples;

public:
	DataParameters(size_t bits_in, size_t bits_out, size_t ones_in = 0,
	               size_t ones_out = 0, size_t samples = 0)
	    : m_bits_in(bits_in),
	      m_bits_out(bits_out),
	      m_ones_in(ones_in),
	      m_ones_out(ones_out),
	      m_samples(samples)
	{
	}

	static DataParameters optimal(const size_t n_bits,
	                              const size_t n_samples = 0,
	                              const size_t n_bits_in = 0,
	                              const size_t n_bits_out = 0);

	bool valid()
	{
		return (m_bits_in > 0) && (m_bits_out > 0) && (m_ones_in > 0) &&
		       (m_ones_out > 0) && (m_samples > 0);
	}

	size_t bits_in() const { return m_bits_in; }
	size_t bits_out() const { return m_bits_out; }
	size_t ones_in() const { return m_ones_in; }
	size_t ones_out() const { return m_ones_out; }
	size_t samples() const { return m_samples; }

	DataParameters &bits_in(size_t bits_in) {
		m_bits_in = bits_in;
		return *this;
	}

	DataParameters &bits_out(size_t bits_out) {
		m_bits_out = bits_out;
		return *this;
	}

	DataParameters &ones_in(size_t ones_in) {
		m_ones_in = ones_in;
		return *this;
	}

	DataParameters &ones_out(size_t ones_out) {
		m_ones_out = ones_out;
		return *this;
	}

	static size_t optimal_sample_count(const DataParameters &params);

	DataParameters &optimal_sample_count()
	{
		m_samples = optimal_sample_count(*this);
		return *this;
	}
};
}

#endif /* CPPNAM_UTIL_PARAMETERS_HPP */