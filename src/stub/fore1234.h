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
#pragma unroll
for (int i0 = 0; i0 < E0; ++i0) {
#pragma unroll
for (int i1 = 0; i1 < E1; ++i1) {
#pragma unroll
for (int i2 = 0; i2 < E2; ++i2) {
#pragma unroll
for (int i3 = 0; i3 < E3; ++i3) {
const int eid = (((i0*E1)+i1)*E2+i2)*E3+i3;
