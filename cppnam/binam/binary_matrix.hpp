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
 * @file binary_matrix.hpp
 *
 * Contains an implementation of a dense binary matrix class in which each entry
 * occupies exactly one bit in memory.
 *
 * @author Christoph Jenzen
 * @author Andreas Stöckel
 */

#pragma once

#ifndef CPPNAM_UTIL_BINARY_MATRIX_HPP
#define CPPNAM_UTIL_BINARY_MATRIX_HPP

#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <vector>

#ifndef NDEBUG
#include <sstream>
#include <stdexcept>
#endif

namespace nam {
/**
 * Integer type to be used internally to store the bit matrix.
 */
using BinaryMatrixCell = uint64_t;

/**
 * Class used to access individual bits as returned by the BinaryMatrix,
 * BinaryVector and BitIterator classes.
 */
class BitReference {
private:
	/**
	 * Pointer at the cell containing the referenced value.
	 */
	BinaryMatrixCell &m_cell;

	/**
	 * Index of the cell.
	 */
	uint8_t m_idx;

public:
	/**
	 * Constructor of the BitReference class.
	 *
	 * @param cell is a reference at the cell that is being represented by the
	 * bit reference.
	 * @param idx is the bit index within that cell.
	 */
	BitReference(BinaryMatrixCell &cell, uint8_t idx) : m_cell(cell), m_idx(idx)
	{
	}

	/**
	 * Returns the value of the bit that is being referenced.
	 *
	 * @return the value of the referenced bit.
	 */
	operator bool() const { return m_cell & (1L << m_idx); }

	/**
	 * Sets the value of the cell to the given value.
	 *
	 * @param b is the value the bit at the given reference should be set to.
	 */
	BitReference &operator=(bool b)
	{
		m_cell = b ? (m_cell | (1L << m_idx)) : (m_cell & ~(1L << m_idx));
		return *this;
	}
};

/**
 * Base class used for both the BitRowIterator and the BitColIterator classes.
 *
 * @tparam Const if true, this is a const iterator (the values this iterator
 * points at cannot be modified).
 */
template <bool Const>
class BitIterator {
public:
	using PtrType = typename std::conditional<Const, BinaryMatrixCell const *,
	                                          BinaryMatrixCell *>::type;

protected:
	/**
	 * Pointer at the cell containing the referenced value.
	 */
	PtrType m_cell;

	/**
	 * Index of the cell.
	 */
	uint8_t m_idx;

public:
	/**
	 * Constructor of the BitIterator class.
	 *
	 * @param cell is a reference at the cell that is being represented by the
	 * bit reference.
	 * @param idx is the bit index within that cell.
	 */
	BitIterator(PtrType cell, uint8_t idx)
	    : m_cell(cell + idx / std::numeric_limits<BinaryMatrixCell>::digits),
	      m_idx(idx % std::numeric_limits<BinaryMatrixCell>::digits)
	{
	}

	template <typename O>
	bool operator==(const O &o) const
	{
		return m_cell == o.m_cell && m_idx == o.m_idx;
	}

	template <typename O>
	bool operator!=(const O &o) const
	{
		return !(*this == o);
	}

	template <typename O>
	bool operator<(const O &o) const
	{
		return m_cell < o.m_cell || (m_cell == o.m_cell && m_idx < o.m_idx);
	}

	template <typename O>
	bool operator>(const O &o) const
	{
		return m_cell > o.m_cell || (m_cell == o.m_cell && m_idx > o.m_idx);
	}

	template <typename O>
	bool operator<=(const O &o) const
	{
		return !(*this > o);
	}

	template <typename O>
	bool operator>=(const O &o) const
	{
		return !(*this < o);
	}

	template <typename U = BinaryMatrixCell,
	          typename = typename std::enable_if<!Const, U>::type>
	BitReference operator*()
	{
		return BitReference(*m_cell, m_idx);
	}

	template <typename U = BinaryMatrixCell,
	          typename = typename std::enable_if<!Const, U>::type>
	BitReference operator->()
	{
		return BitReference(*m_cell, m_idx);
	}

	bool operator*() const
	{
		return BitReference(const_cast<BinaryMatrixCell &>(*m_cell), m_idx);
	}

	bool operator->() const
	{
		return BitReference(const_cast<BinaryMatrixCell &>(*m_cell), m_idx);
	}
};

namespace internal {
/**
 * Iterator which allows to iterate over the values of a row.
 */
template <bool Const>
class GenericBitRowIterator : public BitIterator<Const> {
public:
	using BitIterator<Const>::BitIterator;
	using BitIterator<Const>::m_cell;
	using BitIterator<Const>::m_idx;

	/**
	 * Returns a const version of the iterator.
	 */
	operator GenericBitRowIterator<true>() const
	{
		return GenericBitRowIterator<true>(m_cell, m_idx);
	}

	GenericBitRowIterator &operator++()
	{
		m_idx++;
		if (m_idx >= std::numeric_limits<BinaryMatrixCell>::digits) {
			m_idx = 0;
			m_cell++;
		}
		return *this;
	}

	GenericBitRowIterator operator++(int)
	{
		GenericBitRowIterator res = *this;
		++(*this);
		return res;
	}

	ptrdiff_t operator-(const GenericBitRowIterator &o) const
	{
		return ptrdiff_t((m_cell - o.m_cell) *
		                 std::numeric_limits<BinaryMatrixCell>::digits) +
		       ptrdiff_t(m_idx) - ptrdiff_t(o.m_idx);
	}

	/**
	 * Returns the content of the current cell the row iterator points at. The
	 * value of the bit index is ignored.
	 */
	BinaryMatrixCell cell() const { return *m_cell; }

	/**
	 * Sets the current cell to the given value. Note that the current bit
	 * position of the iterator is ignored.
	 */
	template <typename U = BinaryMatrixCell,
	          typename = typename std::enable_if<!Const, U>::type>
	void cell(BinaryMatrixCell c)
	{
		*m_cell = c;
	}

	/**
	 * Skips to the next cell. Does not alter the value of the bit index this
	 * iterator points at.
	 */
	void next_cell() { m_cell++; }
};

/**
 * Iterator which allows to iterate over the values of a column.
 */
template <bool Const>
class GenericBitColIterator : public BitIterator<Const> {
private:
	size_t m_stride;

public:
	using PtrType = typename BitIterator<Const>::PtrType;
	using BitIterator<Const>::m_cell;
	using BitIterator<Const>::m_idx;

	GenericBitColIterator(PtrType cell, uint8_t idx, size_t stride)
	    : BitIterator<Const>(cell, idx), m_stride(stride)
	{
	}

	/**
	 * Returns a const version of the iterator.
	 */
	operator GenericBitColIterator<true>() const
	{
		return GenericBitColIterator<true>(m_cell, m_idx, m_stride);
	}

	GenericBitColIterator &operator++()
	{
		m_cell += m_stride;
		return *this;
	}

	GenericBitColIterator operator++(int)
	{
		GenericBitColIterator res = *this;
		++(*this);
		return res;
	}

	ptrdiff_t operator-(const GenericBitColIterator &o) const
	{
		return (m_cell - o.m_cell) / m_stride;
	}
};
}

using BitRowIterator = internal::GenericBitRowIterator<false>;
using ConstBitRowIterator = internal::GenericBitRowIterator<true>;
using BitColIterator = internal::GenericBitColIterator<false>;
using ConstBitColIterator = internal::GenericBitColIterator<true>;

/**
 * Type which may be used in the initialization of a BinaryMatrix instance.
 */
template <size_t Rows, size_t Cols>
using BinaryArray = std::array<std::array<bool, Cols>, Rows>;

/**
 * Matrix class used for the storage of the BiNAM connection matrix and binary
 * input patterns. This class uses bit-wise manipulation to access individual
 * bits in memory.
 */
class BinaryMatrix {
private:
	/**
	 * Number of bits in the used integer type.
	 */
	static constexpr size_t INT_WIDTH =
	    std::numeric_limits<BinaryMatrixCell>::digits;

	/**
	 * Maximum number that can be represented by the integer type -- equivalent
	 * to an integer with all bits set to one.
	 */
	static constexpr BinaryMatrixCell INT_MAX =
	    std::numeric_limits<BinaryMatrixCell>::max();

	/**
	 * Calculates the number of IntType instances needed to represent n bits.
	 *
	 * @param n is the number of bits that need to be represented.
	 * @return the number of IntType instances needed to represent the given bit
	 * number.
	 */
	static constexpr size_t number_of_cells(size_t n)
	{
		return ((n + INT_WIDTH - 1) & ~(INT_WIDTH - 1)) / INT_WIDTH;
	};

	/**
	 * Returns the cell in which the given bit is stored.
	 *
	 * @param n is the bit index
	 */
	static constexpr size_t cell_idx(size_t n) { return n / INT_WIDTH; };

	/**
	 * Returns the index of the bit index withoin
	 */
	static constexpr size_t bit_idx(size_t n) { return n & (INT_WIDTH - 1); };

	/**
	 * Vector holding the memory
	 */
	std::vector<BinaryMatrixCell> m_cells;

	/**
	 * Number of rows and number of columns.
	 */
	size_t m_rows, m_cols;

#ifndef NDEBUG
	void check_range(size_t row, size_t col) const;
	void check_range_row(size_t row) const;
	void check_range_col(size_t col) const;
#else /* NDEBUG */
	void check_range(size_t, size_t) const {}
	void check_range_row(size_t) const {}
	void check_range_col(size_t) const {}
#endif /* NDEBUG */

	/**
	 * Returns the cell index the given bit is stored in.
	 *
	 * @param row is the index of the row from which the bit should be read.
	 * @param col is the index of the column from which the bit should be read.
	 * @return the index of the cell in which the given
	 */
	size_t cell_idx(size_t row, size_t col) const
	{
		return row * number_of_cells(m_cols) + cell_idx(col);
	}

public:
	/**
	 * Default constructor. Creates an empty matrix.
	 */
	BinaryMatrix() : m_rows(0), m_cols(0){};

	/**
	 * Initialises the matrix with zeros.
	 */
	BinaryMatrix(size_t rows, size_t cols)
	    : m_cells(rows * number_of_cells(cols)), m_rows(rows), m_cols(cols)
	{
	}

	/**
	 * 2D array constructor.
	 */
	template <size_t Rows, size_t Cols>
	BinaryMatrix(const BinaryArray<Rows, Cols> &init)
	    : BinaryMatrix(Rows, Cols)
	{
		for (size_t i = 0; i < Rows; i++) {
			for (size_t j = 0; j < Cols; j++) {
				(*this)(i, j) = init[i][j];
			}
		}
	}

	/**
	 * Returns a reference at the bit in the given row and column.
	 *
	 * @param row is the index of the row from which the bit should be read.
	 * @param col is the index of the column from which the bit should be read.
	 * @return a reference at the specified bit.
	 */
	BitReference operator()(size_t row, size_t col)
	{
		check_range(row, col);
		return BitReference(m_cells[cell_idx(row, col)], bit_idx(col));
	}

	/**
	 * Returns the value of the bit at the given position.
	 *
	 * @param row is the index of the row from which the bit should be read.
	 * @param col is the index of the column from which the bit should be read.
	 * @return a constant reference at the specified bit.
	 */
	bool operator()(size_t row, size_t col) const
	{
		check_range(row, col);
		return BitReference(
		    const_cast<BinaryMatrixCell &>(m_cells[cell_idx(row, col)]),
		    bit_idx(col));
	}

	/**
	 * Returns an iterator at the beginning of the specified row in the matrix.
	 */
	BitRowIterator begin_row(size_t row)
	{
		check_range_row(row);
		return BitRowIterator(&m_cells[cell_idx(row, 0)], 0);
	}

	/**
	 * Returns an iterator at the end of the specified row in the matrix.
	 */
	BitRowIterator end_row(size_t row)
	{
		check_range_row(row);
		return BitRowIterator(&m_cells[cell_idx(row, m_cols)], bit_idx(m_cols));
	}

	/**
	 * Returns a constant iterator which points at the beginning of the
	 * specified row in the matrix.
	 */
	ConstBitRowIterator begin_row(size_t row) const
	{
		return const_cast<BinaryMatrix *>(this)->begin_row(row);
	}

	/**
	 * Returns a constant iterator which points at the end of the specified row
	 * in the matrix.
	 */
	ConstBitRowIterator end_row(size_t row) const
	{
		return const_cast<BinaryMatrix *>(this)->end_row(row);
	}

	/**
	 * Returns a column-wise iterator which points at the beginning of a column
	 * in the matrix.
	 */
	BitColIterator begin_col(size_t col)
	{
		check_range_col(col);
		return BitColIterator(&m_cells[cell_idx(0, col)], bit_idx(col),
		                      number_of_cells(m_cols));
	}

	/**
	 * Returns a column-wise iterator which points at the end of a column in the
	 * matrix.
	 */
	BitColIterator end_col(size_t col)
	{
		check_range_col(col);
		return BitColIterator(&m_cells[cell_idx(m_rows, col)], bit_idx(col),
		                      number_of_cells(m_cols));
	}

	/**
	 * Returns a constant, column-wise iterator which points at the beginning of
	 * a column in the matrix.
	 */
	ConstBitColIterator begin_col(size_t col) const
	{
		return const_cast<BinaryMatrix *>(this)->begin_col(col);
	}

	/**
	 * Returns a constant, column-wise iterator which points at the end of a
	 * column in the matrix.
	 */
	ConstBitColIterator end_col(size_t col) const
	{
		return const_cast<BinaryMatrix *>(this)->end_col(col);
	}

	/**
	 * Returns the total number of bits stored in the matrix.
	 */
	size_t size() const { return m_rows * m_cols; }

	/**
	 * Returns the total number of rows stored in the matrix.
	 */
	size_t rows() const { return m_rows; }

	/**
	 * Returns the total number of columns stored in the matrix.
	 */
	size_t cols() const { return m_cols; }

	/**
	 * Returns true if this matrix equals the other.
	 */
	bool operator==(const BinaryMatrix &o) const
	{
		if (cols() != o.cols() || rows() != o.rows()) {
			return false;
		}
		return m_cells == o.m_cells;
	}

	/**
	 * Returns true if this matrix equals the other.
	 */
	bool operator!=(const BinaryMatrix &o) const
	{
		if (cols() != o.cols() || rows() != o.rows()) {
			return true;
		}
		return m_cells != o.m_cells;
	}
};

/**
 * Variant of the BinaryMatrix class containing exactly one row and an
 * additional array access operator.
 */
class BinaryVector : public BinaryMatrix {
public:
	/**
	 * Default constructor. Creates an empty vector.
	 */
	BinaryVector() : BinaryMatrix() {};

	/**
	 * Initialises the matrix with zeros.
	 */
	BinaryVector(size_t size) : BinaryMatrix(1, size)
	{
	}

	/**
	 * Initialises a vector with the given initializer list.
	 */
	BinaryVector(std::initializer_list<bool> init) : BinaryVector(init.size())
	{
		size_t i = 0;
		for (bool b: init) {
			(*this)[i++] = b;
		}
	}

	BitReference operator[](size_t i) { return (*this)(0, i); }
	bool operator[](size_t i) const { return (*this)(0, i); }
};

/**
 * Prints a matrix for testing purposes.
 */
static inline std::ostream &operator<<(std::ostream &os, const BinaryMatrix &m)
{
	for (size_t i = 0; i < m.rows(); i++) {
		const auto end = m.end_row(i);
		for (auto it = m.begin_row(i); it != end; it++) {
			os << int(*it);
		}
		os << "\n";
	}
	return os;
}
}

#endif /* CPPNAM_UTIL_BINARY_MATRIX_HPP */
