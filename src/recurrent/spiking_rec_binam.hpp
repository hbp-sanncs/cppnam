/*
 *  CppNAM -- C++ Neural Associative Memory Simulator
 *  Copyright (C) 2016  Christoph Jenzen
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

#ifndef CPPNAM_RECURRENT_SPIKING_REC_BINAM_HPP
#define CPPNAM_RECURRENT_SPIKING_REC_BINAM_HPP

#include <memory>

#include <cypress/cypress.hpp>
#include "core/spiking_netw_basis.hpp"
#include "core/parameters.hpp"
#include "core/spiking_parameters.hpp"
#include "recurrent/rec_binam.hpp"

namespace nam {
/**
 * This is the class implementation of the spiking binam with recurrent
 * connections. Containing all necessary parameter structures, it is building,
 * executing and evaluating cypress networks. It is fed by a simple JSON
 * structure.
 */
class SpikingRecBinam : public SpNetwBasis {
private:
	cypress::Network m_net;
	cypress::Population<cypress::SpikeSourceArray> m_pop_source;
	cypress::PopulationBase m_pop_output;
	std::string m_neuronType;
	DataParameters m_dataParams;
	NeuronParameters m_neuronParams;
	NetworkParameters m_networkParams;
	std::shared_ptr<RecBinam> m_recBinam;

public:
	/**
	 * Constructor: Takes a simple Json and reads in all the parameters.
	 * It sets up the RecBinam, runs and evaluates them (latter one for
	 * comparison).
	 */
	SpikingRecBinam(cypress::Json &json, std::ostream &out = std::cout,
	                bool recall = true);

	/**
	 * Constructor, which overwites DataParameters from JSON
	 */
	SpikingRecBinam(cypress::Json &json, DataParameters params,
	                std::ostream &out = std::cout, bool recall = true,
	                bool read = false);

	/**
	 * Constructor, which overwites DataParameters from JSON and uses external
	 * DataGenerationParameters
	 */
	SpikingRecBinam(cypress::Json &json, DataParameters params,
	                DataGenerationParameters gen_params,
	                std::ostream &out = std::cout, bool recall = true,
	                bool warn = false);

	~SpikingRecBinam() override{};

	std::unique_ptr<SpNetwBasis> clone() override
	{
		return std::make_unique<SpikingRecBinam>(*this);
	};

	/**
	 * Complete building of the spiking neural network
	 */
	SpikingRecBinam &build(cypress::Network &network) override;
	SpikingRecBinam &build() override { return build(m_net); };

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
	void recall() override { m_recBinam->recall(); };
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
};
}

#endif /* CPPNAM_RECURRENT_SPIKING_REC_BINAM_HPP */
