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

#ifndef CPPNAM_CORE_BINAM_HPP
#define CPPNAM_CORE_BINAM_HPP
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "core/entropy.hpp"
#include "core/parameters.hpp"
#include "util/binary_matrix.hpp"
#include "util/data.hpp"
#include "util/population_count.hpp"

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
			if (out.get_bit(i)) {
				for (size_t j = 0; j < Base::numberOfCells(Base::cols()); j++)
					Base::set_cell(i, j, Base::get_cell(i, j) | in.get_cell(j));
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
	BiNAM(size_t output, size_t input) : BinaryMatrix<T>(output, input){};

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
		}
		return *this;
	}

	/**
	 * Sum of all set bits of a BinaryVector. Used for recall
	 */
	size_t digit_sum(const BinaryVector<T> vec)
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
	BinaryVector<T> recall(BinaryVector<T> in, size_t thresh)
	{
		BinaryVector<T> vec(Base::rows());
		BinaryVector<T> temp = in;
		// size_t thresh = digit_sum(in);
		for (size_t i = 0; i < Base::rows(); i++) {
			size_t sum = digit_sum(temp.VectorMult(Base::row_vec(i)));
			if (sum >= thresh) {
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

	BinaryMatrix<T> recallMat(BinaryMatrix<T> in, size_t thresh)
	{
		if (in.cols() != Base::cols()) {
			std::stringstream ss;
			ss << in.size() << " out of range for matrix of size "
			   << Base::cols() << std::endl;
			throw std::out_of_range(ss.str());
		}
		BinaryMatrix<T> res(in.rows(), Base::rows());
		for (size_t i = 0; i < res.rows(); i++) {
			res.write_vec(i, recall(in.row_vec(i), thresh));
		};
		return res;
	}

	/**
	 * Calculation of false positives and negative for single sample
	 * @param out is the original sample
	 * @param recall the one with errors (the recalled sample)
	 */
	static SampleError false_bits(BinaryVector<T> out, BinaryVector<T> recall)
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
	static std::vector<SampleError> false_bits_mat(BinaryMatrix<T> out,
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
	DataGenerationParameters m_datagen;
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
	BiNAM_Container(DataParameters params, DataGenerationParameters datagen)
	    : m_BiNAM(params.bits_out(), params.bits_in()),
	      m_params(params),
	      m_datagen(datagen){};
	BiNAM_Container(DataParameters params)
	    : m_BiNAM(params.bits_out(), params.bits_in()),
	      m_params(params),
	      m_datagen(){};
	BiNAM_Container(){};

	/**
	 * Generates input and output data, trains the storage matrix
	 */
	BiNAM_Container<T> &set_up()
	{
		size_t seed =
		    m_datagen.seed() ? m_datagen.seed() : std::random_device()();

		std::thread input_thread([this, seed]() mutable {
			m_input = DataGenerator(seed, m_datagen.random(),
			                        m_datagen.balanced(), m_datagen.unique())
			              .generate<T>(m_params.bits_in(), m_params.ones_in(),
			                           m_params.samples());
		});
		std::thread output_thread([this, seed]() mutable {
			m_output =
			    DataGenerator(seed + 5, m_datagen.random(),
			                  m_datagen.balanced(), m_datagen.unique())
			        .generate<T>(m_params.bits_out(), m_params.ones_out(),
			                     m_params.samples());
		});

		input_thread.join();
		output_thread.join();

		m_BiNAM.train_mat(m_input, m_output);
		return *this;
	};

	BiNAM_Container<T> &set_up_from_file()
	{
		std::cout << "Read in data-file..." << std::endl;
		size_t height = 0;
		size_t width = 0;
		std::fstream ss("../data/data_in", std::fstream::in);
		if (!ss.good()) {
			throw;
		}
		ss.read((char *)&width, sizeof(width));
		ss.read((char *)&height, sizeof(height));
		m_input = BinaryMatrix<T>(height, width);
		ss.read((char *)m_input.cells().data(),
		        m_input.cells().size() * sizeof(T));
		ss.close();
		if (m_input.cols() != m_params.bits_in() ||
		    m_input.rows() != m_params.samples()) {
			std::stringstream s;
			s << "Input data size " << m_input.cols() << " and "
			  << m_input.rows() << " differs from given Parameters "
			  << m_params.bits_in() << " and " << m_params.samples() << " !"
			  << std::endl;
			throw std::out_of_range(s.str());
		}

		ss.open("../data/data_out", std::fstream::in);
		if (!ss.good()) {
			throw;
		}
		ss.read((char *)&width, sizeof(width));
		ss.read((char *)&height, sizeof(height));
		m_output = BinaryMatrix<T>(height, width);
		ss.read((char *)m_output.cells().data(),
		        m_output.cells().size() * sizeof(T));
		ss.close();
		if (m_output.cols() != m_params.bits_out() ||
		    m_output.rows() != m_params.samples()) {
			std::stringstream s;
			s << "Output data size " << m_output.cols() << " and "
			  << m_output.rows() << " differs from given Parameters "
			  << m_params.bits_out() << " and " << m_params.samples() << " !"
			  << std::endl;
			throw std::out_of_range(s.str());
		}
		std::cout << "\t\t...done" << std::endl;
		m_BiNAM.train_mat(m_input, m_output);
		return *this;
	}

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
	static SampleError sum_false_bits(std::vector<SampleError> vec_err)
	{
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
	 * Calculate the false positives and negatives as well as the stored
	 * information.
	 * @param recall_matrix: recalled matrix from experiment. If nothing is
	 * given, it takes the member recall matrix.
	 */
	ExpResults analysis(
	    const BinaryMatrix<T> &recall_matrix = BinaryMatrix<T>())
	{
		auto recall_mat = &recall_matrix;
		if (recall_matrix.size() == 0) {
			recall_mat = &m_recall;
		}
		std::vector<SampleError> se =
		    m_BiNAM.false_bits_mat(m_output, *recall_mat);
		double info = entropy_hetero(m_params, se);
		SampleError sum = sum_false_bits(se);
		return ExpResults(info, sum);
	};

	/**
	 * Getter for member matrices
	 */
	const BiNAM<T> &trained_matrix() const { return m_BiNAM; };
	const BinaryMatrix<T> &input_matrix() const { return m_input; };
	const BinaryMatrix<T> &output_matrix() const { return m_output; };
	const BinaryMatrix<T> &recall_matrix() const { return m_recall; };

	void trained_matrix(BiNAM<T> mat) { m_BiNAM = mat; };
	void input_matrix(BinaryMatrix<T> mat) { m_input = mat; };
	void output_matrix(BinaryMatrix<T> mat) { m_output = mat; };
	void recall_matrix(BinaryMatrix<T> mat) { m_recall = mat; };

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

#endif /* CPPNAM_CORE_BINAM_HPP */
