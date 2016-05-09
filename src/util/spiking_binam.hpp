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

#ifndef CPPNAM_UTIL_SPIKING_BINAM_HPP
#define CPPNAM_UTIL_SPIKING_BINAM_HPP

#include <array>
#include <string>

#include "binam.hpp"

#include <cypress/cypress.hpp>
#include <cypress/util/json.hpp>
#include <cypress/util/matrix.hpp>

#include "spiking_parameters.hpp"

namespace nam {
/**
 * This is the class implementation of the spiking binam. Containing all
 * necessary parameter structures, it is building, executing and evaluating
 * cypress networks. It is fed by a simple JSON structure.
 */
class SpikingBinam {
private:
	BiNAM_Container<uint64_t> m_BiNAM_Container;
	NeuronParameters m_neuronParams;
	NetworkParameters m_networkParams;
	DataParameters m_dataParams;
	cypress::Network m_net;
	std::string m_neuronType;
	cypress::Population<cypress::SpikeSourceArray> m_pop_source;
	cypress::PopulationBase m_pop_output;

	/**
	 * Creates a population of type @param T and adds them to m_net
	 */
	template <typename T>
	cypress::PopulationBase add_typed_population();

	/**
	 * Runs add_typed_population, but gets a string containing the neuron type
	 * instead of a template argument
	 */
	cypress::PopulationBase add_population(std::string neuron_type_str);

	/**
	 * Build the spike times for the spike source array using the input matrix
	 * in BiNAM_Container and the corresponding parameters
	 */
	std::vector<std::vector<float>> build_spike_times();

	/**
	 * Converts the spike times to an output matrix which can be compared to the
	 * output patterns
	 */
	BinaryMatrix<uint64_t> spikes_to_matrix();

public:
	/**
	 * Constructor: Takes a simple Json and reads in all the parameters.
	 * It sets up the BiNAM_Container, runs and evaluates them (latter one for
	 * comparison).
	 */
	SpikingBinam(cypress::Json &json);

	/**
	 * Getters for the parameter structures.
	 */
	const NetworkParameters &NetParams() const { return m_networkParams; }
	const DataParameters &DataParams() const { return m_dataParams; }
	const NeuronParameters &NeuronParams() const { return m_neuronParams; }

	/**
	 * Complete building of the spiking neural network
	 */
	SpikingBinam& build();

	/**
	 * Execution on hardware or software platform, where @param backend is the
	 * repective platform
	 */
	void run(std::string backend);

	/**
	 * Evaluation based on that one used in the BiNAM_Container
	 */
	void eval_output();
};
}

#endif /* CPPNAM_UTIL_SPIKING_BINAM_HPP */