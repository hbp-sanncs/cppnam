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

#ifndef CPPNAM_UTIL_BINARY_MATRIX_HPP
#define CPPNAM_UTIL_BINARY_MATRIX_HPP

#include <iostream>
#include <limits>
#include <vector>

#ifndef NDBUG
#include <sstream>
#include <stdexcept>
#endif

#include <cypress/util/matrix.hpp>

namespace nam {

using cypress::Matrix;
using cypress::MatrixFlags;
using cypress::Vector;

template <typename T>
class BinaryMatrix;

template <typename T>
class BinaryVector;

/**
 * Matrix class which is used for the BiNAM and storage of patterns.
 * Rows are stored in integer types, which are given as template type.
 * Then we use bit-wise manipulation. Change in values are done by set_bit and
 * set_cell. Cells mean whole integer types. Via set_bit and get_bit one can use
 * the simple row/column numbers as one would do in a normal matrix
 */
template <typename T>
class BinaryMatrix {
public:
	/**
	 * Number of bits in the used integer type.
	 */
	static constexpr uint32_t intWidth = std::numeric_limits<T>::digits;

	/**
	 * Maximum number that can be represented by the integer type -- equivalent
	 * to an integer with all bits set to one.
	 */
	static constexpr T intMax = std::numeric_limits<T>::max();

	/**
	 * Calculates the number of IntType instances needed to represent n bits.
	 *
	 * @param n is the number of bits that need to be represented.
	 * @return the number of IntType instances needed to represent the given bit
	 * number.
	 */
	static constexpr uint32_t numberOfCells(uint32_t n)
	{

		return ((n + intWidth - 1) & ~(intWidth - 1)) / intWidth;
	};

	static constexpr uint32_t cellNumber(uint32_t n)
	{
		return int(n / intWidth);
	};

private:
	/**
	* Shared pointer referencing the memory, used for the copy on write
	* behaviour.
	*/
	Matrix<T> m_mat;

	/**
	 * Number of rows and number of columns.
	 */
	size_t m_rows, m_cols;

public:
	/**
	 * Default constructor. Creates an empty matrix.
	 */
	BinaryMatrix() : m_rows(0), m_cols(0){};

	/**
	 * Initialiser with zeros.
	 */
	BinaryMatrix(uint32_t rows, uint32_t cols) : m_rows(rows), m_cols(cols)
	{
		m_mat = Matrix<T>(rows, numberOfCells(cols), MatrixFlags::ZEROS);
	};

#ifndef NDEBUG
	/**
	 * Check if bit-numbers are in range of matrix to avoid overflows
	 */
	void check_range(uint32_t row, uint32_t col) const
	{
		if (row >= m_rows || col >= m_cols) {
			std::stringstream ss;
			ss << "[" << row << ", " << col
			   << "] out of range for matrix of size " << m_rows << " x "
			   << m_cols << std::endl;
			throw std::out_of_range(ss.str());
		}
	}
	/**
	 * Check if cell-numbers are in range of matrix to avoid overflows
	 */
	void check_range_cells(uint32_t row, uint32_t col) const
	{
		if (row >= m_rows || col >= numberOfCells(m_cols)) {
			std::stringstream ss;
			ss << "[" << row << ", " << col
			   << "] out of range for matrix of size " << m_rows << " x "
			   << numberOfCells(m_cols) << std::endl;
			throw std::out_of_range(ss.str());
		}
	}
#else
	/**
	 * Check if bit-numbers are in range of matrix to avoid overflows
	 */
	void check_range(uint32_t, uint32_t) const {}
	/**
	 * Check if cell-numbers are in range of matrix to avoid overflows
	 */
	void check_range_cells(uint32_t, uint32_t) const {}

#endif
	/**
	 * Read a bit at [row,col]
	 */
	bool get_bit(uint32_t row, uint32_t col) const
	{
		check_range(row, col);
		uint32_t m = col % intWidth;
		return m_mat(row, cellNumber(col)) & (T(1) << m);
	}

	/**
	 * Set a bit at [row,col]
	 */
	BinaryMatrix<T> &set_bit(uint32_t row, uint32_t col, bool val = true)
	{
		check_range(row, col);
		uint32_t m = col % intWidth;
		if (val) {
			m_mat(row, cellNumber(col)) |= (T(1) << m);
		} else {
			m_mat(row, cellNumber(col)) &= ~(T(1) << m);
		}
		return *this;
	}

	/**
	 * Get a cell at [row,cell-col]
	 */
	T get_cell(uint32_t row, uint32_t col) const
	{
		check_range_cells(row, col);
		return m_mat(row, col);
	}

	/**
	 * Set a cell at [row,cell-col]
	 */
	BinaryMatrix<T> &set_cell(uint32_t row, uint32_t col, T value)
	{
		check_range_cells(row, col);
		m_mat(row, col) = value;
		return *this;
	}
	
	/**
	 * Return data matrix
	 */
	Matrix<T> cells(){
		return m_mat;
	}

	/**
	 * Give out matrix sizes
	 */
	size_t size() const { return m_rows * m_cols; };
	size_t rows() const { return m_rows; };
	size_t cols() const { return m_cols; };

	/**
	 * Gives back the row @param i as BinaryVector
	 */
	BinaryVector<T> row_vec(size_t i)
	{
		BinaryVector<T> vec(m_cols);
		for (size_t j = 0; j < numberOfCells(m_cols); j++) {

			vec.set_cell(j, get_cell(i, j));
		}
		return vec;
	}

	/**
	 * Write a whole row from @param vector.
	 * Checks if dimension of vector and matrix are the same.
	 */
	void write_vec(size_t row, BinaryVector<T> vec)
	{
		if (row >= m_rows || m_cols != vec.size()) {
			std::stringstream ss;
			ss << "Either row " << row << " out of bounds of " << m_rows
			   << " or wrong vector size:" << vec.size() << " , " << m_cols
			   << std::endl;
			throw std::out_of_range(ss.str());
		}
		check_range(m_rows - 1, vec.size() - 1);
		vec.check_range(0, m_cols - 1);
		for (size_t i = 0; i < numberOfCells(vec.size()); i++) {
			m_mat(row, i) = vec.get_cell(i);
		}
	}

	void write_col_vec(size_t col, Vector<uint8_t> vec)
	{
		if (col >= m_cols || m_rows != vec.size()) {
			std::stringstream ss;
			ss << "Either row " << col << " out of bounds of " << m_cols
			   << " or wrong vector size:" << vec.size() << " , " << m_rows
			   << std::endl;
			throw std::out_of_range(ss.str());
		}
		check_range(vec.size() - 1, col);
		for (size_t i = 0; i < vec.size(); i++) {
			if (vec(i) > 0) {
				set_bit(i, col);
			}
		}
	}

	/**
	 * Gives back a 'normal' matrix
	 */
	Matrix<uint8_t> convertToMatrix()
	{
		Matrix<uint8_t> mat(m_rows, m_cols);
		for (size_t i = 0; i < m_rows; i++) {
			for (size_t j = 0; j < m_cols; j++) {
				mat(i, j) = int(get_bit(i, j));
			}
		}
		return mat;
	}

	/**
	 * Print the matrix for testing purposes
	 */
	void print() const
	{
		for (size_t i = 0; i < m_rows; i++) {
			for (size_t j = 0; j < m_cols; j++) {
				std::cout << int(get_bit(i, j));
			}
			std::cout << "\n";
		}
		std::cout << std::endl;
	}
};

/**
 * Binary vector class, compound of integers of type @param T. Usage is similar
 * to BinaryMatrix class
 */
template <typename T>
class BinaryVector : public BinaryMatrix<T> {
public:
	using Base = BinaryMatrix<T>;

	BinaryVector() : BinaryMatrix<T>() {}
	BinaryVector(uint32_t size) : BinaryMatrix<T>(1, size) {}

	/**
	 * Functions for manipulating the vector
	 */
	T get_cell(uint32_t row) const { return Base::get_cell(0, row); }

	bool get_bit(uint32_t row) const { return Base::get_bit(0, row); }

	BinaryVector<T> &set_bit(uint32_t row)
	{
		Base::set_bit(0, row);
		return *this;
	}

	void set_cell(uint32_t row, T value) { Base::set_cell(0, row, value); }

	/**
	 * Component-wise multiplication of two vectors. NOT a scalar-product
	 */
	BinaryVector<T> VectorMult(BinaryVector<T> b)
	{
		if (Base::size() != b.size()) {
			std::stringstream ss;
			ss << "Vector-multiplication with dimensions:" << Base::size()
			   << " and " << b.size() << "not possible!" << std::endl;
			throw std::out_of_range(ss.str());
		}
		BinaryVector<T> vec(Base::size());
		for (size_t i = 0; i < Base::numberOfCells(Base::size()); i++) {
			vec.set_cell(i, Base::get_cell(0, i) & b.get_cell(i));
		}
		return vec;
	}
};

/**
 * Expressions for the linker: Workaround for gtest to compile
 */
template <typename T>
constexpr T BinaryMatrix<T>::intMax;

template <typename T>
constexpr uint32_t BinaryMatrix<T>::intWidth;
}

#endif /* CPPNAM_UTIL_BINARY_MATRIX_HPP */

