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

#ifndef CPPNAM_CORE_SPIKE_TRAINS_HPP
#define CPPNAM_CORE_SPIKE_TRAINS_HPP

#include "spiking_parameters.hpp"
#include "util/binary_matrix.hpp"

namespace nam {
/**
 * Buils a spike train
 * @param net_params contains all parameters like standard deviations for
 * jitter,...
 * @param value is the value represented by the spike train (0 or 1)
 * @param offs is the general offset added to all spike times
 * @param return: a vector of variable length containing spike times
 */
std::vector<float> build_spike_train(NetworkParameters net_params,
                                     bool value = true, float offs = 0.0,
                                     int seed = -1);

/**
 * Uses the output of a neuron to calculate the output pattern.
 * @param spikes:  vector of spike times of a single neuron
 * @param samples: number of samples represented by the spike times
 * @param return: the resulting vector containing 0,1
 */
Vector<uint8_t> spikes_to_vector(Matrix<float> spikes, size_t samples,
                                 const NetworkParameters params);

Vector<uint8_t> spikes_to_vector_tresh(Matrix<float> spikes, size_t samples,
                                       const NetworkParameters params);
}

#endif /* CPPNAM_CORE_SPIKE_TRAINS_HPP */
