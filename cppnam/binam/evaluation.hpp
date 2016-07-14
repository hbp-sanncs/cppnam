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

/**
 * @file evaluation.hpp
 *
 * Contains functions and data structures used to evaluate the results from a
 * BiNAM.
 *
 * @author Andreas Stöckel
 */

#pragma once

#ifndef CPPNAM_BINAM_EVALUATION_HPP
#define CPPNAM_BINAM_EVALUATION_HPP

#include <cppnam/binam/population_count.hpp>

namespace nam {
/**
 * The DataParameters structure holds the parameters for a BiNAM with uniformly
 * generated data. These parameters are used throughout the evaluation
 * functions.
 */
class DataParameters {
private:
	size_t m_bits_in;
	size_t m_bits_out;
	size_t m_ones_in;
	size_t m_ones_out;
	size_t m_samples;

public:
	/**
	 * Creates a new DataParameters instance. Allows to directly set the members
	 * of the class. The class also allows to initialize the members explicitly
	 * using chained function calls.
	 *
	 * @param bits_in is the dimensionality of the input vectors.
	 * @param bits_out is the dimensionality of the output vectors.
	 * @param ones_in is the number of bits set to one in each input vector.
	 * @param ones_out is the number of bits set to one in each output vector.
	 */
	DataParameters(size_t bits_in = 0, size_t bits_out = 0, size_t ones_in = 0,
	               size_t ones_out = 0, size_t samples = 0)
	    : m_bits_in(bits_in),
	      m_bits_out(bits_out),
	      m_ones_in(ones_in),
	      m_ones_out(ones_out),
	      m_samples(samples)
	{
	}

	/**
	 * Calculates the number of samples for the given data parameters for which
	 * the largest entropy can be reached.
	 *
	 * @param params are the data parameters for which the optimal sample count
	 * should be calculated.
	 * @return the optimal number of samples.
	 */
	static size_t optimal_sample_count(const DataParameters &params);

	/**
	 * Sets the sample count of this DataParameters instance to the optimal
	 * sample count.
	 *
	 * @return a reference to this DataParameters instance for member function
	 * chaining.
	 */
	DataParameters &optimal_sample_count()
	{
		m_samples = optimal_sample_count(*this);
		return *this;
	}

	/**
	 * Generates optimal data parameters for the given input-/output dimension.
	 * If a sample count is given, that data parameters are calculated for a
	 * fixed sample count.
	 *
	 * @param bits is the number of input-/output bits.
	 * @param samples is the optional number of samples for which the data
	 * parameters should be optimized.
	 * @return a new DataParameters with all data parameters set.
	 */
	static DataParameters optimal(const size_t bits, const size_t samples = 0);

	/**
	 * Copies the input and output dimensions and number of ones to each other
	 * if one of these is not set.
	 *
	 * @return a reference to this DataParameters instance for member function
	 * chaining.
	 */
	DataParameters &canonicalize();

	/**
	 * Checks whether all data parameter entries are set to true.
	 *
	 * @return true if all values are set to something that is not zero.
	 */
	bool valid()
	{
		return (m_bits_in > 0) && (m_bits_out > 0) && (m_ones_in > 0) &&
		       (m_ones_out > 0) && (m_samples > 0);
	}

	size_t bits_in() const { return m_bits_in; }
	size_t bits_out() const { return m_bits_out; }
	size_t ones_in() const { return m_ones_in; }
	size_t ones_out() const { return m_ones_out; }
	size_t samples() const { return m_samples; }

	DataParameters &bits_in(size_t bits_in)
	{
		m_bits_in = bits_in;
		return *this;
	}

	DataParameters &bits_out(size_t bits_out)
	{
		m_bits_out = bits_out;
		return *this;
	}

	DataParameters &ones_in(size_t ones_in)
	{
		m_ones_in = ones_in;
		return *this;
	}

	DataParameters &ones_out(size_t ones_out)
	{
		m_ones_out = ones_out;
		return *this;
	}

	DataParameters &samples(size_t samples)
	{
		m_samples = samples;
		return *this;
	}
};

/**
 * Datastructure containing the number of false positives and false negatives
 * per sample.
 */
struct SampleError {
	double fp, fn;

	SampleError(double fp = 0.0, double fn = 0.0) : fp(fp), fn(fn) {}
};

/**
 * Compares two matrices of binary vectors. Returns the number of false
 * positives (the number of bits set in the "recall" vector but not in the
 * "expected" vector), and the number of false negatives (the number of bits set
 * in the "expected" vector but not in the "recall" vector. Note that the recall
 * matrix may contain fewer rows than the
 */
std::vector<SampleError> binary_vector_diff(
    const BinaryMatrix &expected, const BinaryMatrix &recall);

/**
 * Calculates the to be expected average number of false positives for the
 * given data parameters.
 *
 * @param params structure containing the data parameters.
 * @return the expected number of false positives per sample.
 */
double expected_false_positives(size_t);

/**
 * Calculates the expected entropy for data with the given parameters. See
 * expected_false_positives for a description.
 *
 * @param params structure containing the data parameters.
 */
double expected_entropy(const DataParameters &params);

/**
 * Calculates the entropy for an estimated number of false positives err (which
 * might be real-valued).
 *
 * @param params structure containing the data parameters.
 * @param errs estimated number of false positives per sample.
 */
double entropy_hetero_uniform(const DataParameters &params,
                              double false_positives);

/**
 * Calculates the entropy from an errors-per sample matrix (returned by
 * analyseSampleErrors) and for the given output vector size and the mean
 * number of set bits. All values may also be real/floating point numbers,
 * a corresponding real version of the underlying binomial coefficient is
 * used.
 *
 * @param errs errs is a vector of SampleError containing "fn" and
 * "fp" entries, where "fn" corresponds to the number of false negatives and
 * "fp" to the number of false positives.
 */
double entropy_hetero(const DataParameters &params,
                      const std::vector<SampleError> &errs);

/**
 * Calculates storage capacity of a conventional MxN ROM holding data with
 * the data specification stored in the DataParameters struct.
 */
double conventional_memory_entropy(const DataParameters &params);
}

#endif /* CPPNAM_BINAM_EVALUATION_HPP */
