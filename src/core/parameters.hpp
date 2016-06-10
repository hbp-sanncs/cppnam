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
#include <iostream>
#include <string>

#include <cypress/util/json.hpp>

namespace nam {
class DataGenerationParameters {
private:
	size_t m_seed;
	bool m_random, m_balanced, m_unique;

public:
	DataGenerationParameters(size_t seed, bool random, bool balanced,
	                         bool unique)
	    : m_seed(seed),
	      m_random(random),
	      m_balanced(balanced),
	      m_unique(unique){};
	DataGenerationParameters(const cypress::Json &obj);
	DataGenerationParameters()
	    : m_seed(0), m_random(true), m_balanced(true), m_unique(true){};

	size_t seed() const { return m_seed; }
	bool random() const { return m_random; }
	bool balanced() const { return m_balanced; }
	bool unique() const { return m_unique; }

	void seed(size_t seed) { m_seed = seed; }
	void random(size_t random) { m_random = random; }
	void balanced(size_t balanced) { m_balanced = balanced; }
	void unique(size_t unique) { m_unique = unique; }

	void print(std::ostream &out = std::cout)
	{
		out << "# Data Generation Parameters" << std::endl;
		out << "Seed: " << m_seed << std::endl
		    << "Random: " << m_random << std::endl
		    << "Balanced: " << m_balanced << std::endl
		    << "Unique: " << m_unique << std::endl;
	}
};

class DataParameters {
private:
	size_t m_bits_in;
	size_t m_bits_out;
	size_t m_ones_in;
	size_t m_ones_out;
	size_t m_samples;

public:
	DataParameters(size_t bits_in = 0, size_t bits_out = 0, size_t ones_in = 0,
	               size_t ones_out = 0, size_t samples = 0)
	    : m_bits_in(bits_in),
	      m_bits_out(bits_out),
	      m_ones_in(ones_in),
	      m_ones_out(ones_out),
	      m_samples(samples)
	{
	}

	DataParameters(const cypress::Json &obj);

	static DataParameters optimal(const size_t bits, const size_t samples = 0);

	DataParameters &canonicalize()
	{
		auto update = [](size_t &a, size_t &b) {
			if (a == 0 && b != 0) {
				a = b;
			}
			else if (a != 0 && b == 0) {
				b = a;
			}
		};
		update(m_bits_in, m_bits_out);
		update(m_ones_in, m_ones_out);
		return *this;
	}

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

	DataParameters &bits_in(size_t bits_in)
	{
		m_bits_in = bits_in;
		return *this;
	}

	DataParameters &bits_out(size_t bits_out)
	{
		m_bits_out = bits_out;
		return *this;
	}

	DataParameters &ones_in(size_t ones_in)
	{
		m_ones_in = ones_in;
		return *this;
	}

	DataParameters &ones_out(size_t ones_out)
	{
		m_ones_out = ones_out;
		return *this;
	}

	DataParameters &samples(size_t samples)
	{
		m_samples = samples;
		return *this;
	}

	static size_t optimal_sample_count(const DataParameters &params);

	DataParameters &optimal_sample_count()
	{
		m_samples = optimal_sample_count(*this);
		return *this;
	}

	DataParameters &set(const std::string name, const size_t value)
	{
		if (name == "n_bits_in") {
			m_bits_in = value;
		}
		else if (name == "n_bits_out") {
			m_bits_out = value;
		}
		else if (name == "n_ones_in") {
			m_ones_in = value;
		}
		else if (name == "n_ones_out") {
			m_ones_out = value;
		}
		else if (name == "n_samples") {
			m_samples = value;
		}
		else {
			throw std::invalid_argument("Unknown parameter \"" + name + "\"");
		}
		return *this;
	}

	void print(std::ostream &out = std::cout)
	{
		out << "# Data Parameters: " << std::endl
		    << "Input Bits: " << m_bits_in << std::endl
		    << "Output Bits: " << m_bits_out << std::endl
		    << "Input Ones: " << m_ones_in << std::endl
		    << "Output Ones: " << m_ones_out << std::endl
		    << "Samples: " << m_samples << std::endl
		    << std::endl;
	}
};
}

#endif /* CPPNAM_UTIL_PARAMETERS_HPP */
