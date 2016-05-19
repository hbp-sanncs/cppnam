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

#ifndef CPPNAM_UTIL_EXPERIMENT_HPP
#define CPPNAM_UTIL_EXPERIMENT_HPP

#include <fstream>
#include <string>
#include <vector>

#include <cypress/cypress.hpp>
#include "spiking_binam.hpp"
#include "util/read_json.hpp"

namespace nam {

class Experiment {
private:
	// Saves the backend where the experiment should run on
	std::string m_backend;

	// Reference to the Json file containing all data
	cypress::Json &json;

	/*
	 * A vector containing all single parameters which should not be swept
	 * over
	 */
	std::vector<std::map<std::string, float>> m_params;

	/*
	 * For every experiment a vector of parameter names is stored. It should be
	 * of the form [structure].[parameter], where [structure] represents the
	 * possible data structures
	 */
	std::vector<std::vector<std::string>> m_sweep_params;

	/*
	 * For every experiment the sweep values are stored. The first index is the
	 * experiment, the second relates to the parameter run and the last belongs
	 * to the indivdual parameter
	 */
	std::vector<std::vector<std::vector<float>>> m_sweep_values;

	/*
	 * List of names of the experiments. Results are saved in files named after
	 * these entries
	 */
	std::vector<std::string> experiment_names;

	/*
	 * A flag which is set if the constructor cannot find any experiment
	 * description which will trigger a "normal" execution of one BiNAM
	 */
	bool standard = false;

	/*
	 * Internal function to start the "normal" execution of a single BiNAM
	 * triggerd by the boolian "standard"
	 */
	void run_standard(std::ostream &out = std::cout);

public:
	Experiment(cypress::Json &json, std::string backend);
	int run(std::ostream &out = std::cout);
};
}
#endif /* CPPNAM_UTIL_EXPERIMENT_HPP */