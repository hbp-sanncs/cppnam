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

#include <atomic>
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
	 * over for every experiment
	 */
	std::vector<std::vector<std::pair<std::string, float>>> m_params;

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

	/**
	 * Vector, which contains the number of repititions for every experiment
	 */
	std::vector<size_t> m_repetitions;

	/*
	 * A flag which is set if the constructor cannot find any experiment
	 * description which will trigger a "normal" execution of one BiNAM
	 */
	bool standard = false;

	/*
	 *  Flag for using the optimal number of samples
	 */
	std::vector<bool> m_optimal_sample;

	/*
	 * Internal function to start the "normal" execution of a single BiNAM
	 * triggerd by the boolian "standard"
	 */
	void run_standard(std::string file_name);
	size_t run_experiment(size_t exp,
	                      std::vector<std::vector<std::string>> &names,
	                      std::ostream &ofs);

public:
	Experiment(cypress::Json &json, std::string backend);
	int run(std::string file_name);

	/**
	 * Tool for reading an experiment description from a Json object.
	 * @param Params will contain name-value pairs of single parameters to set
	 * @param sweep_params all names of sweep parameters,
	 * @param sweep_values all related values
	 * @param repetitions number of repetitions of each experiment
	 * @param optimal_sample_count vector of flags if sample count should be
	 * optimised
	 */
	static void read_in_exp_descr(
	    cypress::Json &json, std::vector<std::pair<std::string, float>> &params,
	    std::vector<std::string> &sweep_params,
	    std::vector<std::vector<float>> &sweep_values,
	    std::vector<size_t> &repetitions,
	    std::vector<bool> &optimal_sample_count);
};

/**
 * SIGINT handler. Sets the global "cancel" flag to true when called once,
 * terminates the program if called twice. This allows to terminate the program,
 * even if it is not responsive (the cancel flag is not checked).
 */
static std::atomic_bool cancel(false);
void int_handler(int);
}
#endif /* CPPNAM_UTIL_EXPERIMENT_HPP */