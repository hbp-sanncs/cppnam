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

#ifndef CPPNAM_UTIL_BINAM_HPP
#define CPPNAM_UTIL_BINAM_HPP
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "data.hpp"
#include "entropy.hpp"
#include "binary_matrix.hpp"
#include "population_count.hpp"

namespace nam {

/**
 * The BiNAM class is the BinaryMatrix class with additional instructions used
 * by the BiNAM_Container. This is basically still a simple matrix class, which
 * knows the concept of training and recalling.
 */
template <typename T>
class BiNAM : public BinaryMatrix<T> {
private:
	/**
	 * Training of a sample pair. Dimensions are not checked, is only for
	 * internal used
	 */
	BiNAM<T> &train_vec(BinaryVector<T> in, BinaryVector<T> out)
	{
		for (size_t i = 0; i < out.size(); i++) {
			if (out.get_bit(i) != 0) {
				for (size_t j = 0; j < Base::numberOfCells(Base::cols()); j++)
					Base::set_cell(i, j, Base::get_cell(i, j) |
					                         (Base::intMax & in.get_cell(j)));
			}
		}
		return *this;
	}

public:
	using Base = BinaryMatrix<T>;
	/**
	 * Constructor - nothing to do here
	 */
	BiNAM(){};
	BiNAM(size_t input, size_t output) : BinaryMatrix<T>(output, input){};

	/**
	 * Training of a sample pair with checking of dimensions
	 */
	BiNAM<T> &train_vec_check(BinaryVector<T> in, BinaryVector<T> out)
	{
		if (in.size() != Base::cols() || out.size() != Base::rows()) {
			std::stringstream ss;
			ss << "[" << in.size() << ", " << out.size()
			   << "] out of range for matrix of size " << Base::cols() << " x "
			   << Base::rows() << std::endl;
			throw std::out_of_range(ss.str());
		}
		return train_vec(in, out);
	}

	/**
	 * Training of whole matrices, should be favoured for using
	 */
	BiNAM<T> &train_mat(BinaryMatrix<T> in, BinaryMatrix<T> out)
	{
		if (in.cols() != Base::cols() || out.cols() != Base::rows() ||
		    in.rows() != out.rows()) {
			std::stringstream ss;
			ss << in.size() << " and " << out.size()
			   << " out of range for matrix of size " << Base::size()
			   << std::endl;
			throw std::out_of_range(ss.str());
		}
		for (size_t i = 0; i < in.rows(); i++) {
			train_vec(in.row_vec(i), out.row_vec(i));
		};
		return *this;
	}

	/**
	 * Sum of all set bits of a BinaryVector. Used for recall
	 */
	size_t digit_sum(BinaryVector<T> vec)
	{
		size_t sum = 0;
		for (size_t i = 0; i < Base::numberOfCells(vec.size()); i++) {
			sum += population_count<T>(vec.get_cell(i));
		};
		return sum;
	}

	/*
	 * Recall procedure for a single sample
	 * @param thresh is the threshold
	 */
	BinaryVector<T> recall(BinaryVector<T> in)
	{
		BinaryVector<T> vec(Base::rows());
		for (size_t i = 0; i < Base::rows(); i++) {
			bool iden = true;
			for (size_t j = 0; j < in.numberOfCells(in.size()); j++) {
				uint64_t v = in.get_cell(j);
				uint64_t w = Base::get_cell(i, j);
				if ((v & w) != v) {
					iden = false;
					break;
				}
			}
			if (iden) {
				vec.set_bit(i);
			}
		}
		return vec;
	};

	/*
	 * Recall procedure for a matrix of samples, @param thresh is the threshold
	 */
	BinaryMatrix<T> recallMat(BinaryMatrix<T> in)
	{
		if (in.cols() != Base::cols()) {
			std::stringstream ss;
			ss << in.size() << " out of range for matrix of size "
			   << Base::cols() << std::endl;
			throw std::out_of_range(ss.str());
		}
		BinaryMatrix<T> res(in.rows(), Base::rows());
		for (size_t i = 0; i < res.rows(); i++) {
			res.write_vec(i, recall(in.row_vec(i)));
		};
		return res;
	}

	/**
	 * Calculation of false positives and negative for single sample
	 * @param out is the original sample
	 * @param recall the one with errors (the recalled sample)
	 */
	SampleError false_bits(BinaryVector<T> out, BinaryVector<T> recall)
	{
		SampleError error;
		for (size_t i = 0; i < Base::numberOfCells(out.size()); i++) {
			T temp = out.get_cell(i) ^ recall.get_cell(i);
			error.fp += population_count<T>(temp & recall.get_cell(i));
			error.fn += population_count<T>(temp & out.get_cell(i));
		}
		return error;
	}

	/**
	 * Calculation of false positives and negative for the matrix
	 * @param out is the original sample matrix
	 * @param recall the one with errors (the recalled one)
	 */
	std::vector<SampleError> false_bits_mat(BinaryMatrix<T> out,
	                                        BinaryMatrix<T> res)
	{
		if (res.rows() > out.rows()) {
			std::stringstream ss;
			ss << res.rows() << " out of range for output matrix of size "
			   << out.rows() << std::endl;
			throw std::out_of_range(ss.str());
		}
		std::vector<SampleError> error(res.rows());
		for (size_t i = 0; i < res.rows(); i++) {
			error[i] = false_bits(out.row_vec(i), res.row_vec(i));
		}
		return error;
	}
};

/**
 * The BiNAM_Container is the general class to evaluate BiNAMs and the easy to
 * use interface.
 * Within the constructor one gives the general descritption of the memory and
 * data-generation. With several commands one can generate the data, train the
 * storage matrix and at last recall and analyse.
 */
template <typename T>
class BiNAM_Container {
public:
	BiNAM<T> m_BiNAM;
	DataParameters m_params;
	DataGenerator m_generator;
	BinaryMatrix<T> m_input, m_output, m_recall;
	std::vector<SampleError> m_SampleError;

public:
	/**
	 * Constructor of the Container. Sets all parameters needed for ongoing
	 * Calculation. Therefore, one representation of the BiNAM_Container class
	 * corresponds exactly to one realisation of associative memory.
	 * @param params contains the BiNAM parameters like network size, number of
	 * samples,...
	 * @param seed for the random number generator which produces the data. Can
	 * be used to generate exactly the same data in consecutive runs.
	 * @param random: flag which (de)activates the randomization of data
	 * @param balanced: activates the algorithm for balanced data generation
	 * @param unique: Suppresses the multiple generation of the same pattern
	 */
	BiNAM_Container(DataParameters params, size_t seed, bool random = true,
	                bool balanced = true, bool unique = true)
	    : m_BiNAM(params.bits_in(), params.bits_out()),
	      m_params(params),
	      m_generator(seed, random, balanced, unique){};
	BiNAM_Container(DataParameters params, bool random = true,
	                bool balanced = true, bool unique = true)
	    : m_BiNAM(params.bits_in(), params.bits_out()),
	      m_params(params),
	      m_generator(random, balanced, unique){};
	BiNAM_Container(){};

	/**
	 * Generates input and output data, trains the storage matrix
	 */
	BiNAM_Container<T> &set_up()
	{
		m_input = m_generator.generate<T>(
		    m_params.bits_in(), m_params.ones_in(), m_params.samples());
		m_output = m_generator.generate<T>(
		    m_params.bits_out(), m_params.ones_out(), m_params.samples());
		m_BiNAM.train_mat(m_input, m_output);
		return *this;
	};

	/**
	 * Recalls the patterns with the input matrix
	 */
	BiNAM_Container<T> &recall()
	{
		m_recall = m_BiNAM.recallMat(m_input);
		m_SampleError = m_BiNAM.false_bits_mat(m_output, m_recall);
		return *this;
	};

	/**
	 * Returns the vector of SampleError containing the number of false
	 * positives and negatives per sample which is calculated by the recall
	 * function.
	 */
	const std::vector<SampleError> &false_bits() { return m_SampleError; };

	/**
	 * Returns the number of all false positives and negatives of the recall.
	 */
	SampleError sum_false_bits(std::vector<SampleError> vec_err = {
	                               SampleError(-1, -1)})
	{
		if (vec_err[0].fp < 0) {
			vec_err = m_SampleError;
		}
		SampleError sum(0, 0);
		for (size_t i = 0; i < vec_err.size(); i++) {
			sum.fp += vec_err[i].fp;
			sum.fn += vec_err[i].fn;
		}
		return sum;
	};

	/*
	 * Gives back an approximate number of expected false positives
	 */
	SampleError theoretical_false_bits()
	{
		SampleError se(expected_false_positives(m_params), 0);
		return se;
	};

	/**
	 * Prints out the results of analysis: Number of FP and FN, Information
	 * count and expected values.
	 */
	void analysis(std::vector<SampleError> vec_err = {SampleError(-1, -1)},
	              std::ostream &output = std::cout)
	{
		std::vector<SampleError> err;
		bool member_flag = false;
		if (vec_err[0].fp < 0) {
			err = m_SampleError;
			member_flag = true;
		}
		else {
			err = vec_err;
		}
		SampleError se = sum_false_bits(err);
		SampleError se_th = theoretical_false_bits();
		double info = entropy_hetero(m_params, err);
		double info_th = expected_entropy(m_params);

		if (member_flag) {
			output << "Result of the non-spiking analysis" << std::endl;
		}
		else {
			output << "Result of the analysis" << std::endl;
		}
		output << "\tInfo \t nInfo \t fp \t fn" << std::endl;
		output << "theor: \t" << info_th << "\t" << 1.00 << "\t"
		       << se_th.fp * m_params.samples() << "\t" << se_th.fn
		       << std::endl;
		output << "exp: \t" << info << "\t" << info / info_th << "\t" << se.fp
		       << "\t" << se.fn << std::endl;
	};
	
	/**
	 * Analysis for parameter sweeps, gives out results for saving in files
	 * IMPORTANT: Compares to recalled matrix instead of predicted values;
	 */
	void analysis_compare_file(std::vector<SampleError> vec_err,
	                           std::ostream &output)
	{
		recall();
		SampleError se = sum_false_bits(vec_err);
		SampleError se_th = sum_false_bits(m_SampleError);
		double info = entropy_hetero(m_params, vec_err);
		double info_th = entropy_hetero(m_params, m_SampleError);
		output << info << "," << info_th << "," << se.fp << "," << se_th.fp
		       << "," << se.fn << "," << se_th.fn << ",";
	}

	/**
	 * Getter for member matrices
	 */
	const BiNAM<T> &trained_matrix() const { return m_BiNAM; };
	const BinaryMatrix<T> &input_matrix() const { return m_input; };
	const BinaryMatrix<T> &output_matrix() const { return m_output; };
	const BinaryMatrix<T> &recall_matrix() const { return m_recall; };

	/**
	 * Print out matrices for testing purposes
	 */
	void print()
	{
		m_BiNAM.print();
		m_input.print();
		m_output.print();
		m_recall.print();
	};
};
}

#endif /* CPPNAM_UTIL_BINAM_HPP */
