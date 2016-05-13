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

#ifndef CPPNAM_UTIL_SPIKING_PARAMETERS_HPP
#define CPPNAM_UTIL_SPIKING_PARAMETERS_HPP

#include <array>
#include <string>

#include <cypress/cypress.hpp>
#include "parameters.hpp"

/**
 * Macro used for defining the getters and setters associated with a parameter
 * value.
 */
#define NAMED_PARAMETER(NAME, IDX)            \
	static constexpr size_t idx_##NAME = IDX; \
	void NAME(float x) { arr[IDX] = x; }     \
	float &NAME() { return arr[IDX]; }       \
	float NAME() const { return arr[IDX]; }

namespace nam {

class NeuronParameters {
private:
	std::vector<float> m_params;

public:
	NeuronParameters(const cypress::NeuronType &type, const cypress::Json &json,
	                 std::ostream &out = std::cout);
	NeuronParameters(){};

	const std::vector<float> &parameter() const { return m_params; };

	template <typename parameter>
	void change_parameter(double value)
	{
		// Security + change correct value
	}
};

class NetworkParameters {
private:
	std::vector<float> arr;

public:
	static const std::vector<std::string> names;

	NAMED_PARAMETER(input_burst_size, 0);
	NAMED_PARAMETER(output_burst_size, 1);
	NAMED_PARAMETER(time_window, 2);
	NAMED_PARAMETER(isi, 3);
	NAMED_PARAMETER(sigma_t, 4);
	NAMED_PARAMETER(sigma_offs, 5);
	NAMED_PARAMETER(p0, 6);
	NAMED_PARAMETER(p1, 7);
	NAMED_PARAMETER(weight, 8);
	NAMED_PARAMETER(general_offset, 9);
	NetworkParameters(const cypress::Json &json, std::ostream &out = std::cout);
	NetworkParameters() : arr{0, 0, 0, 0, 0, 0, 0, 0, 0, 0} {};
};
}



#undef NAMED_PARAMETER
#endif /* CPPNAM_UTIL_SPIKING_PARAMETERS_HPP */
