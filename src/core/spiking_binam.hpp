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
#include <memory>
#include <string>

#include "binam.hpp"

#include <cypress/cypress.hpp>
#include "spiking_parameters.hpp"
#include "parameters.hpp"

namespace nam {
/**
 * This is the class implementation of the spiking binam. Containing all
 * necessary parameter structures, it is building, executing and evaluating
 * cypress networks. It is fed by a simple JSON structure.
 */
class SpikingBinam {
private:
	cypress::Network m_net;
	cypress::Population<cypress::SpikeSourceArray> m_pop_source;
	cypress::PopulationBase m_pop_output;
	DataParameters m_dataParams;
	NeuronParameters m_neuronParams;
	NetworkParameters m_networkParams;
	std::shared_ptr<BiNAM_Container<uint64_t>> m_BiNAM_Container;

	std::string m_neuronType;

	/**
	 * Creates a population of type @param T and adds them to m_net
	 */
	template <typename T>
	cypress::PopulationBase add_typed_population(cypress::Network &network);

	/**
	 * Runs add_typed_population, but gets a string containing the neuron type
	 * instead of a template argument
	 */
	cypress::PopulationBase add_population(std::string neuron_type_str,
	                                       cypress::Network &network);

	/**
	 * Build the spike times for the spike source array using the input matrix
	 * in BiNAM_Container and the corresponding parameters
	 */
	std::vector<std::vector<float>> build_spike_times(int seed = -1);

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
	SpikingBinam(cypress::Json &json, std::ostream &out = std::cout,
	             bool recall = true);

	/**
	 * Constructor, which overwites DataParameters from JSON
	 */
	SpikingBinam(cypress::Json &json, DataParameters params,
	             std::ostream &out = std::cout, bool recall = true);

	/**
	 * Getters for the parameter structures.
	 */
	const NetworkParameters &NetParams() const { return m_networkParams; }
	const DataParameters &DataParams() const { return m_dataParams; }
	const NeuronParameters &NeuronParams() const { return m_neuronParams; }

	/**
	 * Setters for the parameter structures.
	 */
	void NetParams(NetworkParameters net) { m_networkParams = net; }
	void DataParams(DataParameters data) { m_dataParams = data; }
	void NeuronParams(NeuronParameters params) { m_neuronParams = params; }

	/**
	 * Recall theoretical BiNAM e.g. when DataParameters have been changed
	 */
	void recall() { m_BiNAM_Container->recall(); }

	/**
	 * Complete building of the spiking neural network
	 */
	SpikingBinam &build();
	SpikingBinam &build(cypress::Network &network);

	/**
	 * Execution on hardware or software platform
	 * @param backend is the repective platform
	 * @param argc and
	 * @param argv are the command line options, this is used for the NMPI
	 * execution on the hbp-collab
	 * @param nmpi If true, NMPI and therefore the hbp-collab is used, else PyNN
	 * is executed directly
	 */
	void run(std::string backend);

	/**
	 * Evaluation based on that one used in the BiNAM_Container
	 */
	void evaluate_neat(std::ostream &output = std::cout);
	void evaluate_csv(std::ostream &output);
	std::pair<ExpResults, ExpResults> evaluate_res();
};
}

#endif /* CPPNAM_UTIL_SPIKING_BINAM_HPP */
