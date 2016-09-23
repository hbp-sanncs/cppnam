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

#ifndef CPPNAM_CORE_SPIKING_NETW_BASIS_HPP
#define CPPNAM_CORE_SPIKING_NETW_BASIS_HPP

#include <cypress/cypress.hpp>
#include "entropy.hpp"
#include "parameters.hpp"
#include "spiking_parameters.hpp"

namespace nam {
/**
 * Here we define a virtual basis class which describes a spiking associative
 * memory
 */
class SpNetwBasis {
public:
	virtual ~SpNetwBasis(){};

	/**
	 * Getters for the parameter structures.
	 */
	virtual const NetworkParameters &NetParams() const = 0;
	virtual const DataParameters &DataParams() const = 0;
	virtual const NeuronParameters &NeuronParams() const = 0;

	/**
	 * Setters for the parameter structures.
	 */
	virtual void NetParams(NetworkParameters net) = 0;
	virtual void DataParams(DataParameters data) = 0;
	virtual void NeuronParams(NeuronParameters params) = 0;

	/**
	 * Recall theoretical BiNAM e.g. when DataParameters have been changed
	 */
	virtual void recall() = 0;

	/**
	 * Clone this object. Example implementation:
	 * return std::make_unique<BiNAM>(*this);
	 */
	virtual std::unique_ptr<SpNetwBasis> clone() = 0;

	/**
	 * Complete building of the spiking neural network
	 */
	virtual SpNetwBasis &build() = 0;
	virtual SpNetwBasis &build(cypress::Network &network) = 0;

	/**
	 * Execution on hardware or software platform
	 * @param backend is the respective platform
	 */
	virtual void run(std::string backend) = 0;

	/**
	 * Evaluation: neat is human readable, csv gives comma separated values and
	 * evaluate_res gives it back for further computation
	 */
	virtual void evaluate_neat(std::ostream &output = std::cout) = 0;
	virtual void evaluate_csv(std::ostream &output) = 0;
	virtual std::pair<ExpResults, ExpResults> evaluate_res() = 0;
};
}

#endif /* CPPNAM_CORE_SPIKING_NETW_BASIS_HPP */
