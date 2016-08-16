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
#include <vector>

#include <cypress/cypress.hpp>
#include "parameters.hpp"
#include "spiking_binam.hpp"
#include "spiking_parameters.hpp"
#include "util/binary_matrix.hpp"
#include "util/read_json.hpp"

namespace nam {
namespace {

std::vector<float> read_neuron_parameters_from_json(
    const cypress::NeuronType &type, const cypress::Json &obj, bool warn = true)
{
	std::map<std::string, float> input = json_to_map<float>(obj);
	// special case if g_leak was given instead of tau_m,
	// But not on spikey, as there g_leak is the standard parameter name
	auto iter = input.find("g_leak");
	if (iter != input.end() &&
	    &type != &cypress::IfFacetsHardware1::inst()) {
		auto iter2 = input.find("cm");
		float cm = 0;
		if (iter2 == input.end()) {
			for (size_t j = 0; j < type.parameter_names.size(); j++) {
				if (type.parameter_names[j] == "cm") {
					cm = type.parameter_defaults[j];
					break;
				}
			}
		}
		else {
			cm = input["cm"];
		}
		input["tau_m"] = cm / input["g_leak"];
		input.erase(iter);
	}

	// special case if tau_m was given instead of g_leak on spikey
	iter = input.find("tau_m");
	if (iter != input.end() &&
	    &type == &cypress::IfFacetsHardware1::inst()) {
		auto iter2 = input.find("cm");
		float cm = 0;
		if (iter2 == input.end()) {
			for (size_t j = 0; j < type.parameter_names.size(); j++) {
				if (type.parameter_names[j] == "cm") {
					cm = type.parameter_defaults[j];
					break;
				}
			}
		}
		else {
			cm = input["cm"];
		}
		input["g_leak"] = cm / input["tau_m"];
		input.erase(iter);
	}

	return read_check<float>(input, type.parameter_names,
	                         type.parameter_defaults, warn);
}
}

NeuronParameters::NeuronParameters(const cypress::NeuronType &type,
                                   const cypress::Json &json, std::ostream &out,
                                   bool warn)
    : m_parameter_names(type.parameter_names)
{
	m_params = read_neuron_parameters_from_json(type, json["params"], warn);
	out << "# Neuron Parameters: " << std::endl;
	for (size_t i = 0; i < m_params.size(); i++) {
		out << type.parameter_names[i] << ": " << m_params[i] << std::endl;
	}
	out << std::endl;
}

const std::vector<std::string> NetworkParameters::names = {"input_burst_size",
                                                           "output_burst_size",
                                                           "time_window",
                                                           "isi",
                                                           "sigma_t",
                                                           "sigma_offs",
                                                           "p0",
                                                           "p1",
                                                           "weight",
                                                           "multiplicity",
                                                           "general_offset"};

const std::vector<float> NetworkParameters::defaults{1, 1, 100, 1, 0,  0,
                                                     0, 0, 0.1, 1, 100};

NetworkParameters::NetworkParameters(const cypress::Json &obj,
                                     std::ostream &out, bool warn)
{
	out << "# NetworkParameters: " << std::endl;
	std::map<std::string, float> input = json_to_map<float>(obj);
	arr = read_check<float>(input, names, defaults, warn);

	for (size_t i = 0; i < names.size(); i++) {
		out << names[i] << ": " << arr[i] << std::endl;
	}
	out << std::endl;
}
}
