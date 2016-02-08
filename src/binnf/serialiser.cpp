/*
 *  CppNAM -- C++ Neural Associative Memory Simulator
 *  Copyright (C) 2016  Andreas Stöckel
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

#include <stdexcept>
#include <sstream>

#include "serialiser.hpp"

namespace nam {
namespace binnf {

using Number = Serialiser::Number;
using NumberType = Serialiser::NumberType;
using Header = Serialiser::Header;
using Block = Serialiser::Block;
using Callback = Serialiser::Callback;

namespace {

/* Helper methods for the calculation of the total block length */

using SizeType = uint32_t;

static constexpr uint32_t BLOCK_START_SEQUENCE = 0x4b636c42;
static constexpr uint32_t BLOCK_END_SEQUENCE = 0x426c634b;
static constexpr SizeType MAX_STR_SIZE = 1024;
static constexpr SizeType SIZE_LEN = sizeof(SizeType);
static constexpr SizeType TYPE_LEN = sizeof(NumberType);
static constexpr SizeType NUMBER_LEN = sizeof(Number);

SizeType str_len(const std::string &s) { return SIZE_LEN + s.size(); }
SizeType header_len(const Header &header)
{
	SizeType res = SIZE_LEN;
	for (auto &name : header.names) {
		res += str_len(name) + TYPE_LEN;
	}
	return res;
}

SizeType matrix_len(const Matrix<Number> &matrix)
{
	return SIZE_LEN + SIZE_LEN + matrix.size() * NUMBER_LEN;
}

SizeType block_len(const std::string &name,
                          const Header &header,
                          const Matrix<Number> &matrix)
{
	return str_len(name) + header_len(header) + matrix_len(matrix);
}

/* Individual data write methods */

template <typename T>
void write(std::ostream &os, const T &t)
{
	os.write((char *)&t, sizeof(T));
}

template <>
void write(std::ostream &os, const std::string &str)
{
	if (str.size() > MAX_STR_SIZE) {
		std::stringstream ss;
		ss << "String exceeds string size limit of " << MAX_STR_SIZE
		   << " bytes.";
		throw std::overflow_error(ss.str());
	}
	write(os, SizeType(str.size()));
	os.write(str.c_str(), str.size());
}

template <>
void write(std::ostream &os, const Matrix<Number> &matrix)
{
	write(os, SizeType(matrix.rows()));
	write(os, SizeType(matrix.cols()));
	os.write((const char *)matrix.data(), matrix.size() * NUMBER_LEN);
}

/* Specialised read methods */

bool synchronise(std::istream &is, uint32_t marker)
{
	uint32_t sync = 0;
	uint8_t c = 0;
	while (is.good() && sync != marker) {
		is.read((char *)&c, 1);
		sync = (sync >> 8) | (c << 24);  // Requires a little endian machine
	}
	return sync == marker;
}

template <typename T>
bool read(std::istream &is, T &res)
{
	is.read((char *)&res, sizeof(T));
	return is.gcount() == sizeof(T);
}

template <>
bool read(std::istream &is, std::string &str)
{
	// Read the string size, make sure it is below the maximum string size
	SizeType size;
	if (!read(is, size)) {
		return false;
	}
	if (size > MAX_STR_SIZE) {
		return false;
	}

	// Read the actual string content
	str.resize(size);
	is.read(&str[0], size);
	return is.gcount() == size;
}

template <>
bool read(std::istream &is, Matrix<Number> &matrix)
{
	// Read the row and column count
	SizeType rows, cols;
	if (!(read(is, rows) && read(is, cols))) {
		return false;
	}

	// Write the data into the matrix
	matrix.resize(rows, cols);
	SizeType total = matrix.size() * NUMBER_LEN;
	is.read((char*)(matrix.data()), total);
	return is.gcount() == total;
}
}

/* Entire block serialisation method */

void Serialiser::serialise(std::ostream &os, const std::string &name,
                           const Header &header,
                           const Matrix<Number> &matrix)
{
	assert(matrix.cols() == header.size());

	write(os, BLOCK_START_SEQUENCE);  // Write the block start mark
	write(os,
	      block_len(name, header, matrix));  //  Write the total block length
	write(os, name);                         // Write the name of the block
	write(os, SizeType(header.size()));      // Write the length of the header
	for (size_t i = 0; i < header.size(); i++) {
		write(os, header.names[i]);  // Write the column name
		write(os, header.types[i]);  // Write the column type
	}
	write(os, matrix);              // Write the actual matrix
	write(os, BLOCK_END_SEQUENCE);  // Write the block end mark
}

void Serialiser::serialise(std::ostream &os, const Block &block)
{
	serialise(os, block.name, block.header, block.matrix);
}

/* Entire block deserialisation method */

std::pair<bool, Block> Serialiser::deserialise(std::istream &is)
{
	Block res;

	// Try to read the block start header
	if (!synchronise(is, BLOCK_START_SEQUENCE)) {
		return std::make_pair(false, Block());
	}

	// Read the block size
	SizeType block_size;  // Note: the block size is currently unused
	if (!read(is, block_size)) {
		return std::make_pair(false, Block());
	}

	// Read the name
	if (!read(is, res.name)) {
		return std::make_pair(false, Block());
	}

	// Read the number of header elements
	SizeType header_count;
	if (!read(is, header_count)) {
		return std::make_pair(false, Block());
	}

	// Read the header elements
	res.header.names.resize(header_count);
	res.header.types.resize(header_count);
	for (size_t i = 0; i < header_count; i++) {
		if (!(read(is, res.header.names[i]) && read(is, res.header.types[i]))) {
			return std::make_pair(false, Block());
		}
	}

	// Read the matrix
	if (!read(is, res.matrix)) {
		return std::make_pair(false, Block());
	}

	// Make sure the block ends with the block end sequence
	uint32_t block_end = 0;
	if (!read(is, block_end) || block_end != BLOCK_END_SEQUENCE) {
		return std::make_pair(false, Block());
	}

	return std::make_pair(true, std::move(res));
}

void Serialiser::deserialise(std::istream &is, const Callback &callback)
{
	std::pair<bool, Block> res;  // Result structure
	do {
		res = deserialise(is);
		if (res.first) {
			res.first =
			    callback(res.second.name, res.second.header, res.second.matrix);
		}
	} while (res.first);
}
}
}