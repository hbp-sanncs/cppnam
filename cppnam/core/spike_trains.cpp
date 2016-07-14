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

#include "spike_trains.hpp"

#include <algorithm>  //std::sort
#include <random>
#include <vector>
#include <iostream>

#include "spiking_parameters.hpp"
#include "util/binary_matrix.hpp"

namespace nam {
std::vector<float> build_spike_train(NetworkParameters net_params, bool value,
                                     float offs, int seed)
{
	std::vector<float> res;
	float p;
	std::default_random_engine generator(seed == -1 ? std::random_device()()
	                                                : seed);

	// Draw actual spike offset
	float offset;
	if (net_params.sigma_offs() > 0) {
		std::normal_distribution<float> distribution(0, net_params.sigma_offs());
		offset = offs + distribution(generator);
	}
	else {
		offset = offs;
	}

	if (value) {
		p = net_params.p0();
	}
	else {
		p = 1.0 - net_params.p1();
	}
	std::uniform_real_distribution<float> uni_dist(0.0, 1.0);
	std::normal_distribution<float> norm_dist(0, net_params.sigma_t());
	for (size_t i = 0; i < net_params.input_burst_size(); i++) {
		if (uni_dist(generator) >= p) {
			float jitter = 0;
			if (net_params.sigma_t() > 0) {
				jitter = norm_dist(generator);
			}
			res.emplace_back<float>(offset + i * net_params.isi() + jitter);
		}
	}
	std::sort(res.begin(), res.end());
	return res;
}

Vector<uint8_t> spikes_to_vector(Matrix<float> spikes, size_t samples,
                                 const NetworkParameters params)
{
	Vector<uint8_t> output(samples, cypress::MatrixFlags::ZEROS);
	for (size_t i = 0; i < samples; i++) {
		for (float j : spikes) {
			if (params.general_offset() + params.time_window() * i<= j &&
			    j < params.general_offset() + params.time_window() * (i + 1)) {
				output[i] += 1;
			}
		}
	}
	return output;
}

Vector<uint8_t> spikes_to_vector_tresh(Matrix<float> spikes, size_t samples,
                                       const NetworkParameters params)
{
	Vector<uint8_t> vec = spikes_to_vector(spikes, samples, params);
	for (size_t i = 0; i < vec.size(); i++) {
		if (vec[i] >= params.output_burst_size()) {
			vec[i] = 1;
		}
		else {
			vec[i] = 0;
		}
	}
	return vec;
}
}