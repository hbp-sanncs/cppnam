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

#include "util/read_json.hpp"

#include "entropy.hpp"
#include "parameters.hpp"
#include "util/optimisation.hpp"

namespace nam {

DataGenerationParameters::DataGenerationParameters(const cypress::Json &obj,
                                                   bool warn)
{
	std::map<std::string, size_t> input = json_to_map<size_t>(obj);
	std::vector<std::string> names = {"seed", "random", "balanced", "unique"};
	std::vector<size_t> default_vals({0, 1, 1, 1});
	auto res = read_check<size_t>(input, names, default_vals, warn);
	m_seed = res[0];
	m_random = false;
	m_balanced = false;
	m_unique = false;
	if (res[1]) {
		m_random = true;
	}
	if (res[2]) {
		m_balanced = true;
	}
	if (res[3]) {
		m_unique = true;
	}
}

DataParameters::DataParameters(const cypress::Json &obj, bool warn)
{
	std::map<std::string, size_t> input = json_to_map<size_t>(obj);
	std::vector<std::string> names = {"n_bits_in", "n_bits_out", "n_ones_in",
	                                  "n_ones_out", "n_samples"};
	auto res = read_check<size_t>(input, names,
	                              std::vector<size_t>(names.size(), 0), warn);

	m_bits_in = res[0];
	m_bits_out = res[1];
	m_ones_in = res[2];
	m_ones_out = res[3];
	m_samples = res[4];
	if (m_ones_in == 0) {
		optimal(m_ones_in, m_samples);
	}
	this->canonicalize();
	if (m_samples == 0) {
		optimal_sample_count();
	}
	if (!valid()) {
		throw("Exception in reading Data Parameters");
	}
}

DataParameters DataParameters::optimal(const size_t bits, const size_t samples)
{
	size_t I_max = 0;
	size_t N_max = 0;

	auto goal_fun = [&I_max, &N_max, &bits, &samples](size_t ones) mutable {
		DataParameters params(bits, bits, ones, ones);
		if (params.samples() == 0) {
			params.optimal_sample_count();
		}
		double I = expected_entropy(params);
		if (I == 0) {  // Quirk to make sure the function is truely unimodal
			I = -ones;
		}
		if (I > I_max) {
			I_max = I;
			N_max = params.samples();
		}
		return -I;
	};

	size_t min = 1, max = std::floor(bits / 2) + 1;
	size_t ones = int(find_minimum_unimodal(goal_fun, min, max));
	return DataParameters(bits, bits, ones, ones, N_max);
}

size_t DataParameters::optimal_sample_count(const DataParameters &params)
{

	double p = 1.0 -
	           double(params.ones_in() * params.ones_out()) /
	               double(params.bits_in() * params.bits_out());
	size_t N_min = 0;
	size_t N_max = std::ceil(std::log(0.1) / std::log(p));
	return find_minimum_unimodal(
	    [&params](int N) {
		    return -expected_entropy(DataParameters(params).samples(N));
		},
	    N_min, N_max, 1.0);
}
}
