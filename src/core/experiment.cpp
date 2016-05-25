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

#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "experiment.hpp"
#include "spiking_binam.hpp"
#include "util/read_json.hpp"

namespace nam {
namespace {
/**
 * Splits a string @param s into parts devided by @param delim and stores the
 * result in @param elems and returns it
 */
std::vector<std::string> &split(const std::string &s, char delim,
                                std::vector<std::string> &elems)
{
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

/**
 * The same as above, but only returning the vector.
 */
std::vector<std::string> split(const std::string &s, char delim)
{
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

/**
 * Adds a sweep parameter to already existing structures.
 * @param key : string containing the name of the parameter
 * @param values : Vector of values which should be swept over
 * @param sweep_params : Vector of names of sweep parameters. @key will be
 * appended
 * @param sweep_elems : two dimensional vector containing all sweep values: The
 * first dimension is the run index, while the second addresses the sweep
 * parameters
 */
static void add_sweep_parameter(const std::string &key,
                                const std::vector<float> &values,
                                std::vector<std::string> &sweep_params,
                                std::vector<std::vector<float>> &sweep_elems)
{
	// Add the sweep key
	sweep_params.emplace_back(key);

	// Copy the old sweep elements
	const std::vector<std::vector<float>> old_sweep_elems = sweep_elems;

	// Fetch some constants
	const size_t n_elems_old = old_sweep_elems.size();
	const size_t n_elems_old_p1 = std::max<size_t>(1, n_elems_old);
	const size_t n_elems = n_elems_old_p1 * values.size();
	const size_t n_dim = sweep_params.size();

	// Resize the elements matrix
	sweep_elems.resize(n_elems);
	std::fill(sweep_elems.begin(), sweep_elems.end(),
	          std::vector<float>(n_dim));

	// Copy the old sweep elements, insert the value for the last column
	for (size_t i = 0; i < values.size(); i++) {
		for (size_t j = 0; j < n_elems_old; j++) {
			std::copy(old_sweep_elems[j].begin(), old_sweep_elems[j].end(),
			          sweep_elems[i * n_elems_old + j].begin());
		}
		for (size_t j = 0; j < n_elems_old_p1; j++) {
			sweep_elems[i * n_elems_old_p1 + j][n_dim - 1] = values[i];
		}
	}
}

/**
 * Sets a parameter in the given @param binam, while @param names should contain
 * the entries 'name of structure' and 'parameter name'. The appropriate
 * value willl be set to @param value
 */
void set_parameter(SpikingBinam &binam, std::vector<std::string> names,
                   float value)
{
	if (names[0] == "params") {
		auto params = binam.NeuronParams();
		binam.NeuronParams(params.set(names[1], value));
	}
	else if (names[0] == "network") {
		auto params = binam.NetParams();
		binam.NetParams(params.set(names[1], value));
	}
	/*else if (names[0] == "data") {
		auto params = binam.DataParams();
		binam.DataParams(params.set(names[1], value));
	}*/
	else {
		throw std::invalid_argument("Unknown parameter \"" + names[0] + "\"");
	}
}
}
Experiment::Experiment(cypress::Json &json, std::string backend)
    : m_backend(backend), json(json)
{
	if (json.find("experiments") == json.end()) {
		standard = true;
	}
	else {
		std::vector<std::string> names = {"min", "max", "count"};
		for (auto i = json["experiments"].begin();
		     i != json["experiments"].end(); i++) {
			// For every experiment read in all parameters and values, then
			// append to member vectors
			std::map<std::string, float> params;
			std::vector<std::string> sweep_params;
			std::vector<std::vector<float>> sweep_values;
			std::string name = i.key();

			for (auto j = json["experiments"][name].begin();
			     j != json["experiments"][name].end(); j++) {
				const cypress::Json val = j.value();

				// See if val is a number (no sweep), an array or an object
				if (val.is_number()) {
					params.insert(std::pair<std::string, float>(j.key(), val));
				}
				else if (val.is_array()) {
					if (val.size() == 1) {
						params.insert(
						    std::pair<std::string, float>(j.key(), val));
					}
					else {
						add_sweep_parameter(j.key(), val, sweep_params,
						                    sweep_values);
					}
				}
				else if (val.is_object()) {
					auto map = json_to_map<float>(val);
					auto range = read_check<float>(map, names,
					                               std::vector<float>{0, 0, 0});
					std::vector<float> values;
					double step = (range[1] - range[0]) / range[2];
					for (size_t k = 0; k < range[2]; k++) {
						values.emplace_back<float>(range[0] + float(k) * step);
					}
					add_sweep_parameter(j.key(), values, sweep_params,
					                    sweep_values);
				}
				else {
					throw std::invalid_argument("Unknown Json value!");
				}
			}
			m_params.emplace_back(params);
			m_sweep_params.emplace_back(sweep_params);
			m_sweep_values.emplace_back(sweep_values);
			experiment_names.emplace_back(name);
		}
	}
}

void Experiment::run_standard()
{
	std::ofstream ofs("data_single_run.txt", std::ofstream::app);
	auto time = std::time(NULL);
	ofs << "#"
	    << " ________________________________________________________"
	    << std::endl
	    << "# "
	    << "Spiking Binam from " << std::ctime(&time) << std::endl;
	SpikingBinam m_SpBinam(json, true, ofs);
	m_SpBinam.build().run(m_backend);
	m_SpBinam.evaluate_neat(ofs);
}

void Experiment::run_no_data(size_t exp,
                             std::vector<std::vector<std::string>> names,
                             std::ostream &ofs)
{
	SpikingBinam sp_binam(json);
	for (size_t j = 0; j < m_sweep_values[exp].size(); j++) {  // all values
		for (size_t k = 0; k < m_sweep_values[exp][j].size();
		     k++) {  // all parameter
			set_parameter(sp_binam, names[k], m_sweep_values[exp][j][k]);
			ofs << m_sweep_values[exp][j][k] << ",";
		}

		sp_binam.build().run(m_backend);
		sp_binam.evaluate_csv(ofs);
		ofs << std::endl;
		std::cout << size_t(100 * float(j + 1) / m_sweep_values[exp].size())
		          << "% done from experiment " << exp + 1 << " of "
		          << m_sweep_params.size() << std::endl;
	}
}

void Experiment::run_data(size_t exp,
                          std::vector<std::vector<std::string>> names,
                          std::ostream &ofs)
{
	DataParameters data_params(json["data"]);
	std::vector<size_t> data_indices, other_indices;
	    for (size_t k = 0; k < names.size(); k++)
	{
		if (names[k][0] != "data") {
			other_indices.emplace_back(k);
		}
		else {
			data_indices.emplace_back(k);
		}
	}
	for (size_t j = 0; j < m_sweep_values[exp].size(); j++) {  // all values
		for (auto k : data_indices) {
			data_params.set(names[k][1], m_sweep_values[exp][j][k]);
		}
		SpikingBinam sp_binam(json,data_params);
		for(auto k : other_indices){
			set_parameter(sp_binam, names[k], m_sweep_values[exp][j][k]);
		}
		for(size_t k=0; k< m_sweep_values[exp][j].size(); k++){
			ofs << m_sweep_values[exp][j][k] << ",";
		}
		sp_binam.build().run(m_backend);
		sp_binam.evaluate_csv(ofs);
		ofs << std::endl;
		std::cout << size_t(100 * float(j + 1) / m_sweep_values[exp].size())
		          << "% done from experiment " << exp + 1 << " of "
		          << m_sweep_params.size() << std::endl;
			
	}
}

// TODO restore m_params functionality
// TODO run several experiments on the hardware
// TODO repeat + average
int Experiment::run(std::ostream &out)
{
	if (standard) {
		run_standard();
		return 0;
	}
	for (size_t i = 0; i < m_sweep_params.size(); i++) {  // for every
		                                                  // experiment

		// Splitting names for usage
		std::vector<std::vector<std::string>> names;
		for (auto j : m_sweep_params[i]) {
			names.emplace_back(split(j, '.'));
		}

		// Check, wether DataParameters was changed to do a non-spiking recall
		// for evaluation
		bool data_changed = false;
		for (size_t k = 0; k < names.size(); k++) {
			if (names[k][0] == "data") {
				data_changed = true;
			}
		}

		// Open file and write first line
		std::ofstream ofs(experiment_names[i] + ".txt", std::ofstream::out);
		ofs << "# ";
		for (auto j : names)
			ofs << j[1] << " , ";
		ofs << "info, info_th, fp, fp_th, fn, fn_th," << std::endl;
		if (!data_changed) {
			run_no_data(i, names, ofs);
		}
		else{
			run_data(i, names, ofs);
		}
	}
	return 0;
}
}
