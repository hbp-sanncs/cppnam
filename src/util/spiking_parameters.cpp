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
#include "binary_matrix.hpp"
#include "parameters.hpp"
#include "read_json.hpp"
#include "spiking_binam.hpp"
#include "spiking_parameters.hpp"

namespace nam {
namespace {

std::vector<float> read_neuron_parameters_from_json(
    const cypress::NeuronType &type, const cypress::Json &obj)
{
	std::map<std::string, float> input = json_to_map<float>(obj);
	return read_check<float>(input, type.parameter_names,
	                         type.parameter_defaults);
}
}

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

const std::vector<std::string> NetworkParameters::names = {
    "input_burst_size", "output_burst_size", "time_window", "isi",
    "sigma_t",          "sigma_offs",        "p0",          "p1",
    "weight",           "general_offset"};

NetworkParameters::NetworkParameters(const cypress::Json &obj,
                                     std::ostream &out)
{
	out << "# NetworkParameters: " << std::endl;
	std::map<std::string, float> input = json_to_map<float>(obj);
	arr = read_check<float>(input, names, std::vector<float>(names.size(),0));

	for (size_t i = 0; i < names.size(); i++) {
		out << names[i] << ": " << arr[i] << std::endl;
	}
	out << std::endl;
}


}
