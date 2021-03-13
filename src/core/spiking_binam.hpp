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

#ifndef CPPNAM_CORE_SPIKING_BINAM_HPP
#define CPPNAM_CORE_SPIKING_BINAM_HPP

#include <cypress/cypress.hpp>

#include <array>
#include <memory>
#include <string>

#include "binam.hpp"
#include "parameters.hpp"
#include "spiking_netw_basis.hpp"
#include "spiking_parameters.hpp"

namespace nam {
/**
 * This is the class implementation of the spiking binam. Containing all
 * necessary parameter structures, it is building, executing and evaluating
 * cypress networks. It is fed by a simple JSON structure.
 */
class SpikingBinam : public SpNetwBasis {
private:
	cypress::Network m_net;
	cypress::Population<cypress::SpikeSourceArray> m_pop_source;
	cypress::PopulationBase m_pop_output;
	DataParameters m_dataParams;
	NeuronParameters m_neuronParams;
	NetworkParameters m_networkParams;
	std::shared_ptr<BiNAM_Container<uint64_t>> m_BiNAM_Container;

	std::string m_neuronType;

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
	             std::ostream &out = std::cout, bool recall = true,
	             bool read = false);

	/**
	 * Constructor, which overwites DataParameters from JSON and uses external
	 * DataGenerationParameters
	 */
	SpikingBinam(cypress::Json &json, DataParameters params,
	             DataGenerationParameters gen_params,
	             std::ostream &out = std::cout, bool recall = true,
	             bool warn = false);
	~SpikingBinam() override = default;

	std::unique_ptr<SpNetwBasis> clone() override
	{
		return std::make_unique<SpikingBinam>(*this);
	};

	/**
	 * Getters for the parameter structures.
	 */
	const NetworkParameters &NetParams() const override
	{
		return m_networkParams;
	}
	const DataParameters &DataParams() const override { return m_dataParams; }
	const NeuronParameters &NeuronParams() const override
	{
		return m_neuronParams;
	}

	/**
	 * Setters for the parameter structures.
	 */
	void NetParams(NetworkParameters net) override { m_networkParams = net; }
	void DataParams(DataParameters data) override { m_dataParams = data; }
	void NeuronParams(NeuronParameters params) override
	{
		m_neuronParams = params;
	}

	/**
	 * Recall theoretical BiNAM e.g. when DataParameters have been changed
	 */
	void recall() override { m_BiNAM_Container->recall(); }

	/**
	 * Complete building of the spiking neural network
	 */
	SpikingBinam &build() override;
	SpikingBinam &build(cypress::Network &network) override;

	/**
	 * Execution on hardware or software platform
	 * @param backend is the repective platform
	 */
	void run(std::string backend) override;

	/**
	 * Evaluation based on that one used in the BiNAM_Container
	 */
	void evaluate_neat(std::ostream &output = std::cout) override;
	void evaluate_csv(std::ostream &output) override;
	std::pair<ExpResults, ExpResults> evaluate_res() override;

	const cypress::PopulationBase &get_pop_output() const
	{
		return m_pop_output;
	}
	const cypress::PopulationBase &get_pop_source() const
	{
		return m_pop_source;
	}

	const std::shared_ptr<BiNAM_Container<uint64_t>> &get_BiNAM() const
	{
		return m_BiNAM_Container;
	}
};
}  // namespace nam

#endif /* CPPNAM_CORE_SPIKING_BINAM_HPP */
