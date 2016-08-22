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

#ifndef CPPNAM_CORE_SPIKING_UTILS_HPP
#define CPPNAM_CORE_SPIKING_UTILS_HPP

#include <cypress/cypress.hpp>
#include "core/parameters.hpp"
#include "core/spiking_parameters.hpp"
#include "util/binary_matrix.hpp"


namespace nam{
	
class SpikingUtils{
public:
	/**
	 * Creates a population of type @param T and adds them to m_net
	 */
	template <typename T>
	static cypress::PopulationBase add_typed_population(
	    cypress::Network &network, DataParameters &dataParams,
	    NetworkParameters &netwParams, NeuronParameters &neuronParams);

	/**
	 * Runs add_typed_population, but gets a string containing the neuron type
	 * instead of a template argument
	 */
	static cypress::PopulationBase add_population(
	    std::string &neuron_type_str, cypress::Network &network,
	    DataParameters &dataParams, NetworkParameters &netwParams,
	    NeuronParameters &neuronParams);

	/**
	 * Build the spike times for the spike source array using the input matrix
	 * in BiNAM_Container and the corresponding parameters
	 */
	static std::vector<std::vector<float>> build_spike_times(
	    const BinaryMatrix<uint64_t> &input_mat, NetworkParameters &netwParams,
	    int seed = -1);

	/**
	 * Converts the spike times to an output matrix which can be compared to the
	 * output patterns
	 */
	static BinaryMatrix<uint64_t> spikes_to_matrix(cypress::PopulationBase &popOutput,
	DataParameters &dataParams, NetworkParameters &netwParams);
	
	/**
	 * Helper to return an instance of a certain neuron type
	 */
	static const cypress::NeuronType &detect_type(std::string neuron_type_str);
};
}

#endif /* CPPNAM_CORE_SPIKING_UTILS_HPP */