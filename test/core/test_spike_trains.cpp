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

#include <cstdint>
#include <vector>
#include <cypress/util/matrix.hpp>
#include <gtest/gtest.h>
#include "core/spike_trains.hpp"
#include "core/spiking_parameters.hpp"
#include "util/binary_matrix.hpp"


namespace nam {

TEST(spiketrain, build_spike_train)
{
	NetworkParameters params;
	params.input_burst_size(1.0);
	params.isi(1.0);
	EXPECT_EQ(0, build_spike_train(params, true)[0]);
	params.input_burst_size(5.0);
	EXPECT_EQ(0, build_spike_train(params, true)[0]);
	EXPECT_EQ(1, build_spike_train(params, true)[1]);
	EXPECT_EQ(2, build_spike_train(params, true)[2]);
	EXPECT_EQ(3, build_spike_train(params, true)[3]);
	EXPECT_EQ(4, build_spike_train(params, true)[4]);
	EXPECT_EQ(104, build_spike_train(params, true, 100)[4]);
}

TEST(spiketrain, spikes_to_vector)
{
	NetworkParameters params;
	params.input_burst_size(1.0);
	params.isi(1.0);
	params.input_burst_size(5.0);
	params.output_burst_size(1.0);
	params.time_window(100);
	using namespace cypress;
	Vector<float> spike_vec(
	    {0.1, 303, 709, 710, 711, 903, 904, 905, 906, 907, 10000});
	auto res = spikes_to_vector(spike_vec, 15, params);
	EXPECT_EQ(1, res[0]);
	EXPECT_EQ(0, res[1]);
	EXPECT_EQ(0, res[2]);
	EXPECT_EQ(1, res[3]);
	EXPECT_EQ(0, res[4]);
	EXPECT_EQ(0, res[5]);
	EXPECT_EQ(0, res[6]);
	EXPECT_EQ(1, res[7]);
	EXPECT_EQ(0, res[8]);
	EXPECT_EQ(1, res[9]);
	EXPECT_EQ(0, res[10]);
	
	params.output_burst_size(3);
	
	res = spikes_to_vector(spike_vec, 15, params);
	EXPECT_EQ(0, res[0]);
	EXPECT_EQ(0, res[1]);
	EXPECT_EQ(0, res[2]);
	EXPECT_EQ(0, res[3]);
	EXPECT_EQ(0, res[4]);
	EXPECT_EQ(0, res[5]);
	EXPECT_EQ(0, res[6]);
	EXPECT_EQ(1, res[7]);
	EXPECT_EQ(0, res[8]);
	EXPECT_EQ(1, res[9]);
	EXPECT_EQ(0, res[10]);
}
}