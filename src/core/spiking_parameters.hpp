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

#ifndef CPPNAM_CORE_SPIKING_PARAMETERS_HPP
#define CPPNAM_CORE_SPIKING_PARAMETERS_HPP

#include <array>
#include <iostream>
#include <string>

#include <cypress/cypress.hpp>
#include "parameters.hpp"

/**
 * Macro used for defining the getters and setters associated with a parameter
 * value.
 */
#define NAMED_PARAMETER(NAME, IDX)            \
	static constexpr size_t idx_##NAME = IDX; \
	void NAME(float x) { arr[IDX] = x; }      \
	float &NAME() { return arr[IDX]; }        \
	float NAME() const { return arr[IDX]; }

namespace nam {

class NeuronParameters {
private:
	std::vector<float> m_params;
	std::vector<std::string> m_parameter_names;

public:
	/**
	 * Construct from Json, give out parameters to @param out
	 */
	NeuronParameters(const cypress::NeuronType &type, const cypress::Json &json,
	                 std::ostream &out = std::cout, bool warn = true);

	NeuronParameters(){};

	const std::vector<float> &parameter() const { return m_params; };

	/**
	 * Set parameter with name @param name to @param value
	 */
	NeuronParameters &set(std::string name, float value)
	{
		for (size_t i = 0; i < m_parameter_names.size(); i++) {
			if (m_parameter_names[i] == name) {
				m_params[i] = value;
				return *this;
			}
		}
		throw std::invalid_argument("Unknown neuron parameter" + name);
	}

	float get(std::string name)
	{
		for (size_t i = 0; i < m_parameter_names.size(); i++) {
			if (m_parameter_names[i] == name) {
				return m_params[i];
			}
		}
		throw std::invalid_argument("Unknown neuron parameter" + name);
	}
	void print(std::ostream &out = std::cout)
	{
		out << "# Neuron Parameters: " << std::endl;
		for (size_t i = 0; i < m_params.size(); i++) {
			out << m_parameter_names[i] << ": " << m_params[i] << std::endl;
		}
	}
};

class NetworkParameters {
private:
	std::vector<float> arr;

public:
	static const std::vector<std::string> names;

	/**
	 * Default values
	 */
	static const std::vector<float> defaults;

	NAMED_PARAMETER(input_burst_size, 0);
	NAMED_PARAMETER(output_burst_size, 1);
	NAMED_PARAMETER(time_window, 2);
	NAMED_PARAMETER(isi, 3);
	NAMED_PARAMETER(sigma_t, 4);
	NAMED_PARAMETER(sigma_offs, 5);
	NAMED_PARAMETER(p0, 6);
	NAMED_PARAMETER(p1, 7);
	NAMED_PARAMETER(weight, 8);
	NAMED_PARAMETER(multiplicity, 9);
	NAMED_PARAMETER(general_offset, 10);

	/**
	 * Construct from Json, give out parameters to @param out
	 */
	NetworkParameters(const cypress::Json &json, std::ostream &out = std::cout,
	                  bool warn = true);

	/**
	 * Empty constructor
	 */
	NetworkParameters() : arr{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} {};

	/**
	 * Set parameter with name @param name to @param value
	 */
	NetworkParameters &set(std::string name, float value)
	{
		for (size_t i = 0; i < names.size(); i++) {
			if (names[i] == name) {
				arr[i] = value;
				return *this;
			}
		}
		throw std::invalid_argument("Unknown neuron parameter" + name);
	}

	void print(std::ostream &out = std::cout)
	{
		out << "# Network Parameters: " << std::endl;
		for (size_t i = 0; i < names.size(); i++) {
			out << names[i] << ": " << arr[i] << std::endl;
		}
	}
};
}

#undef NAMED_PARAMETER
#endif /* CPPNAM_CORE_SPIKING_PARAMETERS_HPP */
