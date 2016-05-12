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
#include <algorithm>  //std::sort
#include <random>
#include <string>

#include <cypress/cypress.hpp>
#include "binary_matrix.hpp"
#include "parameters.hpp"
#include "spiking_binam.hpp"

namespace nam {
namespace {
std::vector<float> read_neuron_parameters_from_json(
    const cypress::NeuronType &type, const cypress::Json &obj)
{
	std::vector<float> res(type.parameter_names.size());
	for (size_t i = 0; i < res.size(); i++) {
		if (obj.find(type.parameter_names[i]) != obj.end()) {
			res[i] = obj[type.parameter_names[i]];
		}
		else {
			res[i] = type.parameter_defaults[i];
		}
	}

	return res;
}

template <typename T>
T read_check(cypress::Json json, std::string name, T default_val)
{
	if (json.find(name) != json.end()) {
		return json[name];
	}
	return default_val;
}
}

const std::array<const char *, 10> NetworkParameters::names = {
    "input_burst_size", "output_burst_size", "time_window", "isi",
    "sigma_t",          "sigma_offs",        "p0",          "p1",
    "weight",           "general_offset"};

NeuronParameters::NeuronParameters(const cypress::NeuronType &type,
                                   const cypress::Json &json, std::ostream &out)
{
	m_params = read_neuron_parameters_from_json(type, json["params"]);
	out << "# Neuron Parameters: " << std::endl;
	for (size_t i = 0; i < m_params.size(); i++) {
		out << type.parameter_names[i] << ": " << m_params[i] << std::endl;
	}
	out << std::endl;
}

NetworkParameters::NetworkParameters(const cypress::Json &json,
                                     std::ostream &out)
{
	out << "# NetworkParameters: " << std::endl;
	for (size_t i = 0; i < names.size(); i++) {
		arr[i] = read_check<double>(json, names[i], 0);
		out << names[i] << ": " << arr[i] << std::endl;
	}
	out << std::endl;
}
}
