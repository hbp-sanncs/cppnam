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

#include "gtest/gtest.h"

#include "core/parameters.hpp"
#include "core/spiking_parameters.hpp"
#include "core/spiking_utils.hpp"
#include "util/binary_matrix.hpp"

#include <fenv.h> 
namespace nam {

TEST(spiking_utils, build_spike_train)
{
	NetworkParameters params;
	params.input_burst_size(1.0);
	params.isi(1.0);
	EXPECT_EQ(0, SpikingUtils::build_spike_train(params, true)[0]);
	params.input_burst_size(5.0);
	EXPECT_EQ(0, SpikingUtils::build_spike_train(params, true)[0]);
	EXPECT_EQ(1, SpikingUtils::build_spike_train(params, true)[1]);
	EXPECT_EQ(2, SpikingUtils::build_spike_train(params, true)[2]);
	EXPECT_EQ(3, SpikingUtils::build_spike_train(params, true)[3]);
	EXPECT_EQ(4, SpikingUtils::build_spike_train(params, true)[4]);
	EXPECT_EQ(104, SpikingUtils::build_spike_train(params, true, 100)[4]);
}

TEST(spiking_utils, spikes_to_vector)
{
	NetworkParameters params;
	params.input_burst_size(1.0);
	params.isi(1.0);
	params.input_burst_size(5.0);
	params.output_burst_size(1.0);
	params.time_window(100);
	params.multiplicity(1);
	params.general_offset(0);
	using namespace cypress;
	Vector<cypress::Real> spike_vec(
	    {0.1, 303, 709, 710, 711, 903, 904, 905, 906, 907, 10000});
	auto res = SpikingUtils::spikes_to_vector_tresh(spike_vec, 15, params);
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

	res = SpikingUtils::spikes_to_vector_tresh(spike_vec, 15, params);
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

TEST(spiking_utils, spike_vectors_to_matrix)
{
    feenableexcept(FE_INVALID | FE_OVERFLOW);
	std::vector<double> neuron1({1.5, 2.0, 2.5, 3.1, 3.6});
	std::vector<double> neuron2({1.5, 2.2, 2.4, 2.9, 3.5});
	std::vector<double> neuron3({1.5, 2.0, 2.5, 2.8, 3.9});
	std::vector<double> neuron4({1.5, 2.0, 2.5, 2.6, 3.6});
	std::vector<double> neuron5({1.5, 2.0, 2.5, 3.1, 3.4});
    std::vector<double> neuron6({4.8});
    std::vector<double> neuron7({1.5, 2.0, 2.5, 3.1, 3.7, 4.0});
    std::vector<double> neuron8({1.5, 4.3});
    std::vector<std::vector<double>> spike_mat;
    spike_mat.push_back(neuron1);
    spike_mat.push_back(neuron2);
    spike_mat.push_back(neuron3);
    spike_mat.push_back(neuron4);
    spike_mat.push_back(neuron5);
    spike_mat.push_back(neuron6);
    spike_mat.push_back(neuron7);
    spike_mat.push_back(neuron8);
    NetworkParameters params;
    params.general_offset(0);
    params.time_window(10);
    auto res = SpikingUtils::spike_vectors_to_matrix(spike_mat, 1,params);
    
    EXPECT_EQ(1, res.get_bit(0,0));
	EXPECT_EQ(1, res.get_bit(0,1));
	EXPECT_EQ(1, res.get_bit(0,2));
	EXPECT_EQ(1, res.get_bit(0,3));
	EXPECT_EQ(1, res.get_bit(0,4));
	EXPECT_EQ(0, res.get_bit(0,5));
	EXPECT_EQ(1, res.get_bit(0,6));
    EXPECT_EQ(0, res.get_bit(0,7));

    
    
}
}
