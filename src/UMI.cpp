// Copyright
// 2016, 2017; Yu-Sheng Lin; johnjohnlys@media.ee.ntu.edu.tw
// 2017; Yi-Lun Liao
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
#include "UMI.h"
#include <tuple>
#include <vector>
#include <cinttypes>

namespace UMI {

using namespace std;

void Int4::Print() const
{
	printf(
		"%" PRId32 " "
		"%" PRId32 " "
		"%" PRId32 " "
		"%" PRId32 "\n",
		data[0], data[1], data[2], data[3]
	);
}

PDims_packed::PDims_packed(
	const NDTuple<ParallelDim> &dims,
	const Int4 &expand
):
	expand_(expand)
{
	int32_t warp_size = 1;
	for (size_t i = 0; i < DIM; ++i) {
		const auto &d = dims[i];
		const int32_t eblock_size = expand[i] * d.block_size;
		const int32_t large_warp_size = d.warp_size * d.warp_shuf;
		totaldims_[i] = d.total_size;
		griddims_[i] = CeilDiv(d.total_size, eblock_size);
		blockdims_[i] = d.block_size;
		warpdims_[i] = d.warp_size;
		warpshufs_[i] = d.warp_shuf;
		large_warpdims_[i] = large_warp_size;
		blockdims_in_warp_[i] = d.block_size / large_warp_size;
		eblockdims_[i] = eblock_size;
		warp_size *= d.warp_size;
		if (not IsPowOf2(d.warp_shuf) or d.block_size % large_warp_size != 0) {
			abort();
		}
	}
	if (warp_size != WARP_SIZE) {
		abort();
	}
}

ADims_packed::ADims_packed(const NDTuple<AccumulationDim> &dims)
{
	for (size_t i = 0; i < DIM; ++i) {
		const auto &d = dims[i];
		totaldims_[i] = d.total_size;
		blockdims_[i] = d.block_size;
		griddims_[i] = CeilDiv(d.total_size, d.block_size);
	}
}

ViewDims_packed::ViewDims_packed(const NDTuple<ViewDim> &dims)
{
	for (size_t i = 0; i < DIM; ++i) {
		const auto &d = dims[i];
		dim_ids_[i] = d.physical_dim_id;
		ofss_[i] = d.ofs;
		strides_[i] = d.stride;
	}
}

PhysicalDims_packed::PhysicalDims_packed(const NDTuple<PhysicalDim> &dims)
{
	for (size_t i = 0; i < DIM; ++i) {
		const auto &d = dims[i];
		sizes_[i] = d.size;
		modes_[i] = d.mode;
	}
}

vector<Int4> ComputeGridIdx(
	const Int4 &grid_size,
	const Int4 &mul
) {
	const int grid_total = grid_size.Prod();
	vector<Int4> table(grid_total);
	Int4 counter(0);
	for (int i = 0; i < grid_total; ++i) {
		table[i] = counter;
		table[i] *= mul;
		counter.IncreaseOne(grid_size);
	}
	return table;
}

vector<Int4> ComputeGridIdx(
	const Int4 &grid_size
) {
	const int grid_total = grid_size.Prod();
	vector<Int4> table(grid_total);
	Int4 counter(0);
	for (int i = 0; i < grid_total; ++i) {
		table[i] = counter;
		counter.IncreaseOne(grid_size);
	}
	return table;
}

tuple<
	vector<Int4> /* grid_ofs        */,
	vector<Int4> /* eblock_ofs      */,
	vector<Int4> /* warp_ofs        */,
	vector<Int4> /* thread_ofs      */,
	vector<Int4> /* accum_block_ofs */,
	vector<Int4> /* accum_step_ofs  */
> PrecomputeIdx(const PDims_packed &p_dims, const ADims_packed &a_dims)
{
	// warp_ofs
	vector<Int4> warp_shuf = ComputeGridIdx(p_dims.warpshufs_);
	vector<Int4> warp_ofs_large = ComputeGridIdx(p_dims.blockdims_in_warp_, p_dims.large_warpdims_);
	vector<Int4> warp_ofs(warp_shuf.size()*warp_ofs_large.size());
	{
	size_t i = 0;
	for (const Int4 &wo: warp_ofs_large) {
		for (const Int4 &ws: warp_shuf) {
			warp_ofs[i] = wo;
			warp_ofs[i] += ws;
			i++;
		}
	}
	}
	// subwarp_idx
	vector<Int4> subwarp_idx = ComputeGridIdx(p_dims.warpdims_);
	for (Int4 &swid: subwarp_idx) {
		swid *= p_dims.warpshufs_;
	}
	return make_tuple(
		ComputeGridIdx(p_dims.griddims_, p_dims.eblockdims_),
		ComputeGridIdx(p_dims.expand_, p_dims.blockdims_),
		move(warp_ofs),
		move(subwarp_idx),
		ComputeGridIdx(a_dims.griddims_, a_dims.blockdims_),
		ComputeGridIdx(a_dims.blockdims_)
	);
}

vector<Int4> PrecomputeMofs(
	const vector<Int4> &idxs,
	const Int4 &dim_ids,
	const Int4 &strides,
	const Int4 &base_mofs
) {
	vector<Int4> mofss(idxs.size());
	for (size_t i = 0; i < idxs.size(); ++i) {
		Int4 idx = idxs[i], &mofs = mofss[i];
		mofs = base_mofs;
		idx *= strides;
		mofs.ShuffleAccumulate(idx, dim_ids);
	}
	return mofss;
}

vector<int> PrecomputeMofsLinear(
	const vector<Int4> &idxs,
	const Int4 &dim_ids,
	const Int4 &strides,
	const Int4 &shapes,
	const Int4 &base_mofs
) {
	vector<int> lmofss(idxs.size());
	for (size_t i = 0; i < idxs.size(); ++i) {
		Int4 idx = idxs[i], mofs = base_mofs;
		idx *= strides;
		mofs.ShuffleAccumulate(idx, dim_ids);
		lmofss[i] = mofs.Unfold(shapes);
	}
	return lmofss;
}

tuple<
	vector<Int4> /* glb_load_warp_mofs   */,
	vector<Int4> /* glb_load_thread_mofs */,
	vector<Int4> /* shm_load_warp_mofs   */,
	vector<Int4> /* shm_load_thread_mofs */
> PrecomputeShmofs(const Int4 &shm_shapes, const Int4 &glb_strides, const Int4 &shm_strides) {
	Int4 shm_load_shape;
	for (int i = DIM-1, n = MIN_LOAD_THREAD; i >= 0; --i) {
		const int32_t l = min<int32_t>(n, shm_shapes[i]);
		shm_load_shape[i] = l;
		n /= l;
	}
	Int4 shm_load_shape_total;
	for (int i = 0; i < DIM; ++i) {
		shm_load_shape_total[i] = CeilDiv(shm_shapes[i], shm_load_shape[i]);
	}
	vector<Int4> glb_load_warp_mofs = ComputeGridIdx(shm_load_shape_total, shm_load_shape);
	vector<Int4> glb_load_thread_mofs = ComputeGridIdx(shm_load_shape);
	vector<Int4> shm_load_warp_mofs = glb_load_warp_mofs;
	vector<Int4> shm_load_thread_mofs = glb_load_thread_mofs;
	for (auto &g: glb_load_warp_mofs) g *= glb_strides;
	for (auto &g: glb_load_thread_mofs) g *= glb_strides;
	for (auto &g: shm_load_warp_mofs) g *= shm_strides;
	for (auto &g: shm_load_thread_mofs) g *= shm_strides;
	return make_tuple(
		move(glb_load_warp_mofs),
		move(glb_load_thread_mofs),
		move(shm_load_warp_mofs),
		move(shm_load_thread_mofs)
	);
}

} // end using namespace UMI
