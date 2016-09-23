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

#include <cypress/cypress.hpp>

#include "util/binary_matrix.hpp"
#include "entropy.hpp"
#include "parameters.hpp"
#include "spike_trains.hpp"
#include "spiking_utils.hpp"

namespace nam {
using namespace cypress;
const NeuronType &SpikingUtils::detect_type(std::string neuron_type_str)
{
	if (neuron_type_str == "IF_cond_exp") {
		return IfCondExp::inst();
	}
	else if (neuron_type_str == "IfFacetsHardware1") {
		return IfFacetsHardware1::inst();
	}
	else if (neuron_type_str == "AdExp") {
		return EifCondExpIsfaIsta::inst();
	}
	throw CypressException("Invalid neuron type \"" + neuron_type_str + "\"");
}

std::vector<std::vector<cypress::Real>> SpikingUtils::build_spike_times(
    const BinaryMatrix<uint64_t> &input_mat, NetworkParameters &netwParams,
    int seed)
{
	// BinaryMatrix<uint64_t> mat = m_BiNAM_Container->input_matrix();
	std::vector<std::vector<cypress::Real>> res;
	for (size_t i = 0; i < input_mat.cols(); i++) {  // over all neruons
		for (size_t k = 0; k < netwParams.multiplicity(); k++) {
			std::vector<cypress::Real> vec;
			for (size_t j = 0; j < input_mat.rows(); j++) {  // over all samples
				auto vec2 = build_spike_train(
				    netwParams, input_mat.get_bit(j, i),
				    netwParams.general_offset() + j * netwParams.time_window(),
				    seed++);
				vec.insert(vec.end(), vec2.begin(), vec2.end());
			}
			res.emplace_back(vec);
		}
	}
	return res;
}

template <typename T>
PopulationBase SpikingUtils::add_typed_population(
    cypress::Network &network, DataParameters &dataParams,
    NetworkParameters &netwParams, NeuronParameters &neuronParams)
{
	using Signals = typename T::Signals;
	using Parameters = typename T::Parameters;
	return network.create_population<T>(
	    dataParams.bits_out() * netwParams.multiplicity(),
	    Parameters(neuronParams.parameter()), Signals().record_spikes());
}

PopulationBase SpikingUtils::add_population(std::string &neuron_type_str,
                                            cypress::Network &network,
                                            DataParameters &dataParams,
                                            NetworkParameters &netwParams,
                                            NeuronParameters &neuronParams)
{
	if (neuron_type_str == "IF_cond_exp") {
		return add_typed_population<IfCondExp>(network, dataParams, netwParams,
		                                       neuronParams);
	}
	else if (neuron_type_str == "IfFacetsHardware1") {
		return add_typed_population<IfFacetsHardware1>(
		    network, dataParams, netwParams, neuronParams);
	}
	else if (neuron_type_str == "AdExp") {
		return add_typed_population<EifCondExpIsfaIsta>(
		    network, dataParams, netwParams, neuronParams);
	}

	throw CypressException("Invalid neuron type \"" + neuron_type_str + "\"");
}

BinaryMatrix<uint64_t> SpikingUtils::spikes_to_matrix(
    cypress::PopulationBase &popOutput, DataParameters &dataParams,
    NetworkParameters &netwParams)
{
	BinaryMatrix<uint64_t> res(dataParams.samples(), dataParams.bits_out());
	size_t multi = netwParams.multiplicity();
	for (size_t i = 0; i < dataParams.bits_out(); i++) {
		Vector<uint8_t> spike_vec(dataParams.samples(),
		                          cypress::MatrixFlags::ZEROS);
		for (size_t j = 0; j < multi; j++) {
			auto spikes = popOutput[i * multi + j].signals().data(0);
			auto temp_vec =
			    spikes_to_vector(spikes, dataParams.samples(), netwParams);
			for (size_t k = 0; k < temp_vec.size(); k++) {
				spike_vec[k] += temp_vec[k];
			}
		}
		for (size_t k = 0; k < spike_vec.size(); k++) {
			if (spike_vec[k] >= netwParams.output_burst_size() * multi) {
				res.set_bit(k, i);
			}
		}
	}
	return res;
}
}