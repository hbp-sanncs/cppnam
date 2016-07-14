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

#include <cppnam/binam/evaluation.hpp>

namespace nam {
/**
* Implementation of Golden section search
* https://en.wikipedia.org/wiki/Golden_section_search
*/
template <typename Function>
static double find_minimum_unimodal(Function f, double a, double b,
                                    double tolerance = 1.0)
{
	static const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
	double c = b - gr * (b - a);
	double d = a + gr * (b - a);
	while (std::abs(c - d) > tolerance) {
		const double fc = f(c);
		const double fd = f(d);
		if (fc < fd) {
			b = d;
			d = c;
			c = b - gr * (b - a);
		}
		else {
			a = c;
			c = d;
			d = a + gr * (b - a);
		}
	}
	return (b + a) / 2.0;
}

/*
 * Class DataGenerationParameters
 */

DataGenerationParameters::DataGenerationParameters(const cypress::Json &obj)
{
	std::map<std::string, size_t> input = json_to_map<size_t>(obj);
	std::vector<std::string> names = {"seed", "random", "balanced", "unique"};
	std::vector<size_t> default_vals({0, 1, 1, 1});
	auto res = read_check<size_t>(input, names, default_vals);
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

DataParameters::DataParameters(const cypress::Json &obj)
{
	std::map<std::string, size_t> input = json_to_map<size_t>(obj);
	std::vector<std::string> names = {"n_bits_in", "n_bits_out", "n_ones_in",
	                                  "n_ones_out", "n_samples"};
	auto res =
	    read_check<size_t>(input, names, std::vector<size_t>(names.size(), 0));

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

DataParameters &DataParameters::canonicalize()
{
	auto update = [](size_t &a, size_t &b) {
		if (a == 0 && b != 0) {
			a = b;
		}
		else if (a != 0 && b == 0) {
			b = a;
		}
	};
	update(m_bits_in, m_bits_out);
	update(m_ones_in, m_ones_out);
	return *this;
}

size_t DataParameters::optimal_sample_count(const DataParameters &params)
{
	double p = 1.0 -
	           double(params.ones_in() * params.ones_out()) /
	               double(params.bits_in() * params.bits_out());
	size_t N_min = 0;
	size_t N_max = std::ceil(std::log(0.1) / std::log(p));
	return find_minimum_unimodal([&params](int N) {
		return -expected_entropy(DataParameters(params).samples(N));
	}, N_min, N_max, 1.0);
}

/*
 * Free functions
 */

std::vector<SampleError> accu_binary_vector_difference(
    const BinaryMatrix &expected, const BinaryMatrix &recall)
{
	if (recall.rows() > expected.rows() || expected.cols() != recall.cols()) {
		throw std::out_of_range("Matrix dimensionality mismatch.");
	}

	std::vector<SampleError> error(recall.rows());
	for (size_t i = 0; i < recall.rows(); i++) {
		auto it_recall = recall.begin_row(i);
		auto it_expected = expected.begin_row(i);
		auto it_expected_end = expected.end_row(i);
		for (auto it = it_expected; it != it_expected_end;
		     it.next_cell(), it_expected.next_cell()) {
			BinaryMatrixCell c = it_recall.cell() ^ it_expected.cell();
			error.fp += population_count(c & it_recall.cell());
			error.fn += population_count(c & it_expected.cell());
		}
		error[i] = false_bits(out.row_vec(i), res.row_vec(i));
	}
	return error;
}

double expected_false_positives(const DataParameters &params)
{
	size_t N = params.samples();
	double p = double(params.ones_in() * params.ones_out()) /
	           double(params.bits_in() * params.bits_out());
	return ((params.bits_out() - params.ones_out()) *
	        std::pow(1.0 - std::pow(1.0 - p, N), params.ones_in()));
}

double expected_entropy(const DataParameters &params)
{
	return entropy_hetero_uniform(params, expected_false_positives(params));
}

double entropy_hetero_uniform(const DataParameters &params,
                              double false_positives)
{
	double res = 0.0;
	for (size_t i = 0; i < params.ones_out(); i++) {
		res += std::log2(double(params.bits_out() - i) /
		                 double(params.ones_out() + false_positives - i));
	}
	return res * params.samples();
}

double entropy_hetero(const DataParameters &params,
                      const std::vector<SampleError> &errs)
{
	double ent = 0.0;
	for (auto &err : errs) {
		if (err.fn > 0) {
			ent +=
			    (lnncrr(params.bits_out(), params.ones_out()) -
			     lnncrr(err.fp + params.ones_out() - err.fn,
			            params.ones_out() - err.fn) -
			     lnncrr(params.bits_out() - err.fp - params.ones_out() + err.fn,
			            err.fn)) /
			    std::log(2.0);
		}
		else {
			for (size_t j = 0; j < params.ones_out(); j++) {
				ent += std::log2(double(params.bits_out() - j) /
				                 double(params.ones_out() + err.fp - j));
			}
		}
	}
	return ent;
}

double conventional_memory_entropy(const DataParameters &params)
{
	return params.bits_in() * lnncrr(params.bits_out(), params.ones_out()) /
	       std::log(2.0);
}
}
