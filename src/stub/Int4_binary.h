// Copyright
// 2016, 2017; Yu-Sheng Lin; johnjohnlys@media.ee.ntu.edu.tw
//
// This file is part of UMI.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
DHFunc inline void operator OP(int32_t rhs) {
#pragma unroll 4
	for (int i = 0; i < 4; ++i) {
		data[i] OP rhs;
	}
}

DHFunc inline void operator OP(const Int4 &rhs) {
#pragma unroll 4
	for (int i = 0; i < 4; ++i) {
		data[i] OP rhs.data[i];
	}
}
