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

#include <algorithm>

#include "entropy.hpp"
#include "parameters.hpp"
#include "util/binary_matrix.hpp"
//#include "spike_trains.hpp"
#include "spiking_utils.hpp"

#include <fenv.h>

#define M_PI 3.14159265358979323846 /* pi */

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

std::vector<std::vector<Real>> SpikingUtils::build_spike_times(
    const BinaryMatrix<uint64_t> &input_mat, NetworkParameters &netwParams,
    int seed)
{
	// BinaryMatrix<uint64_t> mat = m_BiNAM_Container->input_matrix();
	std::vector<std::vector<Real>> res;
	for (size_t i = 0; i < input_mat.cols(); i++) {  // over all neruons
		for (size_t k = 0; k < netwParams.multiplicity(); k++) {
			std::vector<Real> vec;
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
    Network &network, DataParameters &dataParams, NetworkParameters &netwParams,
    NeuronParameters &neuronParams)
{
	using Signals = typename T::Signals;
	using Parameters = typename T::Parameters;
	return network.create_population<T>(
	    dataParams.bits_out() * netwParams.multiplicity(),
	    Parameters(neuronParams.parameter()), Signals().record_spikes());
}

PopulationBase SpikingUtils::add_population(std::string &neuron_type_str,
                                            Network &network,
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
    PopulationBase &popOutput, DataParameters &dataParams,
    NetworkParameters &netwParams)
{
	BinaryMatrix<uint64_t> res(dataParams.samples(), dataParams.bits_out());
	size_t multi = netwParams.multiplicity();
	for (size_t i = 0; i < dataParams.bits_out(); i++) {
		Vector<uint8_t> spike_vec(dataParams.samples(), MatrixFlags::ZEROS);
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

std::vector<Real> SpikingUtils::build_spike_train(NetworkParameters net_params,
                                                  bool value, Real offs,
                                                  int seed)
{
	std::vector<Real> res;
	Real p;
	std::default_random_engine generator(seed == -1 ? std::random_device()()
	                                                : seed);

	// Draw actual spike offset
	Real offset;
	if (net_params.sigma_offs() > 0) {
		std::normal_distribution<Real> distribution(0, net_params.sigma_offs());
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
	std::uniform_real_distribution<Real> uni_dist(0.0, 1.0);
	std::normal_distribution<Real> norm_dist(0, net_params.sigma_t());
	for (size_t i = 0; i < net_params.input_burst_size(); i++) {
		if (uni_dist(generator) >= p) {
			Real jitter = 0;
			if (net_params.sigma_t() > 0) {
				jitter = norm_dist(generator);
			}
			res.emplace_back<Real>(offset + i * net_params.isi() + jitter);
		}
	}
	std::sort(res.begin(), res.end());
	return res;
}

Vector<uint8_t> SpikingUtils::spikes_to_vector(Matrix<Real> &spikes,
                                               size_t samples,
                                               const NetworkParameters params)
{
	Vector<uint8_t> output(samples, MatrixFlags::ZEROS);
	for (size_t i = 0; i < samples; i++) {
		for (auto j : spikes) {
			if (params.general_offset() + params.time_window() * i <= j &&
			    j < params.general_offset() + params.time_window() * (i + 1)) {
				output[i] += 1;
			}
		}
	}
	return output;
}

Vector<uint8_t> SpikingUtils::spikes_to_vector_tresh(
    Matrix<Real> &spikes, size_t samples, const NetworkParameters params)
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

Real smoothing_function(Real x)
{
	/*if(x< M_PI && x>0){
	    return std::sin(x)*0.5;// /(0.5*M_PI);
	}
	return 0;*/
	if (-0.1125 < x && x < 0.1125) {
		Real a = std::exp(-1.0 / (1.0 - 64.0 * x * x));
		if (a != a) {
			std::cout << "smoothing_function for " << x << std::endl;
		}
		return 2.25 * std::exp(-1.0 / (1.0 - 64.0 * x * x));
	}
	return 0;
}

Real bin_function(Real x, Real &bin_width, const std::vector<Real> &sample_bins)
{
	if (x >= bin_width * sample_bins.size()) {
		return 0.0;
	}
	return sample_bins[std::floor(x / bin_width)];
}

Real convolution(Real x, Real bin_width, const std::vector<Real> &sample_bins,
                 Real end)
{
	// Real integrator_step = 0.1;
	Real t_max = bin_width * sample_bins.size();
	if (x > t_max) {
		return 0.0;
	}
	Real res =
	    (bin_function(0, bin_width, sample_bins) * smoothing_function(x) +
	     bin_function(end, bin_width, sample_bins) *
	         smoothing_function(x - end)) *
	    0.5;

	for (size_t i = 1; i < floor(end / Real(SpikingUtils::integrator_step)) - 1;
	     i++) {
		res += bin_function(i * SpikingUtils::integrator_step, bin_width,
		                    sample_bins) *
		       smoothing_function(x - i * SpikingUtils::integrator_step);
	}

	return res * SpikingUtils::integrator_step;
}

std::vector<std::vector<Real>> pop_to_spike_vector(PopulationBase &pop)
{
	std::vector<std::vector<Real>> res;
	for (size_t i = 0; i < pop.size(); i++) {
		auto spikes = pop[i].signals().data(0);
		std::vector<Real> temp;
		for (auto spike : spikes) {
			temp.push_back(spike);
		}
		res.push_back(temp);
	}
	return res;
}

// TODO: At the moment no multiplicity
BinaryMatrix<uint64_t> SpikingUtils::spike_vectors_to_matrix(
    std::vector<std::vector<Real>> &spike_mat, size_t samples,
    NetworkParameters &params)
{
	std::vector<std::vector<Real>> bins(samples, std::vector<Real>({0}));
	size_t offset = params.general_offset();
	for (size_t i = 0; i < spike_mat.size(); i++) {  // Neurons
		auto &spikes = spike_mat[i];
		for (size_t j = 0; j < spikes.size(); j++) {
			Real temp = spikes[j] - offset;
			size_t sample_num = std::floor(temp / params.time_window());
			temp = temp - sample_num * params.time_window();
			size_t bin_num = std::floor(temp / bin_width);
			if (bin_num >= bins[sample_num].size()) {
				bins[sample_num].resize(bin_num + 1);
			}
			bins[sample_num][bin_num]++;
		}
	}

	for (auto &vec : bins) {
		Real max = 0.0;
		for (auto val : vec) {
			max += val;
		}
		std::cout << "MAX: " << max << std::endl;
		if (max > 0) {
			for (Real &val : vec) {
				val = val / max;
			}
		}
	}

	std::vector<Real> end_times(samples);
	for (size_t i = 0; i < samples; i++) {
		bool least_reached = false;
		for (size_t j = 0; j < floor(params.time_window() / integrator_step);
		     j++) {
			Real res = convolution(Real(j) * integrator_step, bin_width,
			                       bins.at(i), params.time_window());
			if (res < 0) {
				std::cout << res << " , " << j * integrator_step << std::endl;
			}
			if (res > 0.015) {
				least_reached = true;
			}
			if (!least_reached) {
				continue;
			}
			if (res < 0.015) {
				end_times[i] = j * integrator_step;
				break;
			}
		}
	}
	for (size_t i = 0; i < samples; i++) {
		end_times[i] = end_times[i] + params.general_offset() +
		               Real(i * params.time_window());
		std::cout << end_times[i] << std::endl;
	}

	BinaryMatrix<uint64_t> res(samples, spike_mat.size());
	for (size_t i = 0; i < spike_mat.size(); i++) {  // Neurons
		auto &spikes = spike_mat[i];
		for (size_t j = 0; j < samples; j++) {  // all samples
			for (auto spike : spikes) {
				if ((end_times[j] - importance_intervall) <= spike &&
				    spike <= end_times[j] + importance_intervall) {
					res.set_bit(j, i);
					break;
				}
				if (spike > end_times[j] + importance_intervall) {
					break;
				}
			}
		}
	}
	return res;
}

BinaryMatrix<uint64_t> SpikingUtils::spike_trains_to_matrix(
    PopulationBase &popOutput, DataParameters &data_params,
    NetworkParameters &params)
{
	std::vector<std::vector<Real>> spike_mat = pop_to_spike_vector(popOutput);
	return spike_vectors_to_matrix(spike_mat, data_params.samples(), params);
}

BinaryMatrix<uint64_t> SpikingUtils::spike_vectors_to_matrix2(
    std::vector<std::vector<Real>> &spike_mat, size_t samples,
    NetworkParameters &params)
{
	std::vector<std::vector<Real>> bins(samples, std::vector<Real>({0}));
	size_t offset = params.general_offset();
	for (size_t i = 0; i < spike_mat.size(); i++) {  // Neurons
		auto &spikes = spike_mat[i];
		for (size_t j = 0; j < spikes.size(); j++) {
			Real temp = spikes[j] - offset;
			size_t sample_num = std::floor(temp / params.time_window());
			temp = temp - sample_num * params.time_window();
			size_t bin_num = std::floor(temp / bin_width);
			if (bin_num >= bins[sample_num].size()) {
				bins[sample_num].resize(bin_num + 1);
			}
			if (sample_num >= samples)
				std::cout << "WRONG SAMPLE NUMBER" << std::endl;
			bins[sample_num][bin_num]++;
		}
	}
	std::cout << "stop" << std::endl;

	std::vector<Real> end_times(samples);
	for (size_t i = 0; i < samples; i++) {
		bool least_reached = false;
		Real max = 0.0, last = 0.0;
		for (size_t j = 0; j < floor(params.time_window() / integrator_step);
		     j++) {
			Real res = convolution(Real(j) * integrator_step, bin_width,
			                       bins.at(i), params.time_window());
			if (res > 0.01) {
				least_reached = true;
			}
			if (!least_reached) {
				continue;
			}
			if (res > last && max == 0.0) {
				last = res;
				continue;
			}
			else {
				max = last;
			}
			if (max > 0) {
				if (res / max < 0.2) {
					end_times[i] = j * integrator_step;
					break;
				}
			}
		}
	}
	for (size_t i = 0; i < samples; i++) {
		end_times[i] = end_times[i] + params.general_offset() +
		               Real(i * params.time_window());
		std::cout << end_times[i] << std::endl;
	}

	BinaryMatrix<uint64_t> res(samples, spike_mat.size());
	for (size_t i = 0; i < spike_mat.size(); i++) {  // Neurons
		auto &spikes = spike_mat[i];
		for (size_t j = 0; j < samples; j++) {  // all samples
			for (auto spike : spikes) {
				if ((end_times[j] - importance_intervall) <= spike &&
				    spike <= end_times[j] + importance_intervall) {
					res.set_bit(j, i);
					break;
				}
				if (spike > end_times[j] + importance_intervall) {
					break;
				}
			}
		}
	}
	return res;
}

BinaryMatrix<uint64_t> SpikingUtils::spike_vectors_to_matrix_no_conv(
    std::vector<std::vector<Real>> &spike_mat, size_t samples,
    NetworkParameters &params)
{
	std::vector<std::vector<size_t>> bins(samples, std::vector<size_t>({0}));
	size_t offset = params.general_offset();
	for (size_t i = 0; i < spike_mat.size(); i++) {  // Neurons
		auto &spikes = spike_mat[i];
		for (size_t j = 0; j < spikes.size(); j++) {
			Real temp = spikes[j] - offset;
			size_t sample_num = std::floor(temp / params.time_window());
			temp = temp - sample_num * params.time_window();
			size_t bin_num = std::floor(temp / bin_width);
			if (bin_num >= bins[sample_num].size()) {
				bins[sample_num].resize(bin_num + 1);
			}
			if (sample_num >= samples)
				std::cout << "WRONG SAMPLE NUMBER" << std::endl;
			bins[sample_num][bin_num]++;
		}
	}

	std::vector<Real> end_times(samples);

	for (size_t i = 0; i < bins.size(); i++) {
		// For every sample find the max
		size_t max = *std::max_element(bins[i].begin(), bins[i].end());
		for (size_t j = 0; j < bins[i].size(); j++) {
			if (Real(bins[i][j]) / Real(max) < 0.2) {
				end_times[i] = (j + 0.5) * bin_width;
			}
		}
	}

	for (size_t i = 0; i < samples; i++) {
		end_times[i] = end_times[i] + params.general_offset() +
		               Real(i * params.time_window());
		std::cout << end_times[i] << std::endl;
	}

	BinaryMatrix<uint64_t> res(samples, spike_mat.size());
	for (size_t i = 0; i < spike_mat.size(); i++) {  // Neurons
		auto &spikes = spike_mat[i];
		for (size_t j = 0; j < samples; j++) {  // all samples
			for (auto spike : spikes) {
				if ((end_times[j] - importance_intervall) <= spike &&
				    spike <= end_times[j] + importance_intervall) {
					res.set_bit(j, i);
					break;
				}
				if (spike > end_times[j] + importance_intervall) {
					break;
				}
			}
		}
	}
	return res;
}
}
