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
#pragma once
#include "UMI.h"
#include <cstdio>
#include <cassert>
#include <algorithm>

namespace UMI {

#define CONSTANT_MEMORY_SIZE 8192
__constant__ int32_t LUT[CONSTANT_MEMORY_SIZE];

using namespace std;
template<int i> struct AnInt {};
template <class T> struct PackUtility;

template <
	uint64_t PtrNum,
	int Ofs,
	int N,
	int Stage,
	class EType,
	class UM
> __device__ inline void FillShm(
	EType *shm,
	const int PBID,
	const int ABID,
	const int TID,
	const int n_load_warp,
	const UM &iums
) {
	const int LWID = TID/MIN_LOAD_THREAD;
	const int LTID = TID%MIN_LOAD_THREAD;
#pragma unroll
	for (int i = Ofs; i < Ofs+N; ++i) {
		const auto &ium = iums[i];
		const int n_ptr = Slice(PtrNum, 4, i);
		Int4 glb_ofs = ium.grid_mofs[PBID];
		if (Stage == 1) {
			glb_ofs += ium.accum_block_mofs[ABID];
		}
		if (LTID < ium.load_thread_effective and LWID < n_load_warp) {
			const int shm_size = ium.shm_boundaries.Sum() + 1;
			Int4 glbshm_base_ofs = glb_ofs;
			glbshm_base_ofs += ium.subwarp_glbshm_mofs[LTID];
			Int4 shmshm_base_ofs = ium.subwarp_shmshm_mofs[LTID];
			for (int j = 0; j < ium.shm_load_num; j += n_load_warp) {
				Int4 glb_cur_ofs = ium.warp_glbshm_mofs[j];
				Int4 shm_cur_ofs = ium.warp_shmshm_mofs[j];
				glb_cur_ofs += glbshm_base_ofs;
				shm_cur_ofs += shmshm_base_ofs;
				bool goutside, soutside;
#pragma unroll
				for (int k = 0; k < n_ptr; ++k) {
					const int32_t ga = glb_cur_ofs.ClampedSum(ium.glb_boundaries, ium.glb_modes, goutside);
					const int32_t sa = shm_cur_ofs.ClampedSum(ium.shm_boundaries, soutside) + ium.shm_ofs[k];
					if (not soutside) {
						shm[sa] = goutside ? ium.pad[k] : ium.ptr[k][ga];
					}
				}
			}
		}
	}
	if (N != 0) {
		__syncthreads();
	}
}

template <class A, class B, class C, class D>
__device__ inline void CallStrategy(A &umi, const B &user_const, const C &in, D &out, const AnInt<0>&)
{
	umi.Pre(user_const, in, out);
}
template <class A, class B, class C, class D>
__device__ inline void CallStrategy(A &umi, const B &user_const, const C &in, D &out, const AnInt<1>&)
{
	umi.Loop(user_const, in, out);
}
template <class A, class B, class C, class D>
__device__ inline void CallStrategy(A &umi, const B &user_const, const C &in, D &out, const AnInt<2>&)
{
	umi.Post(user_const, in, out);
}

template <
	uint64_t Expand,
	uint64_t PtrNum,
	uint64_t Opt,
	int Stage,
	int OFS_IN,
	int OFS_OUT,
	int N_IN,
	int N_OUT,
	int TOTAL_IN,
	class EType,
	class CType,
	class IUM,
	class OUM,
	class Umi,
	int li, int lo, int le
> __device__ inline void CoreExecution(
	EType* (&glb)[li],
	EType* (&shm)[lo],
	const IUM &iums,
	const OUM &oums,
	const int abid,
	const int asid,
	const bool (&mvalid)[le],
	Umi (&umi_ctx)[le],
	const CType &uc
) {
	constexpr int E0 = Slice(Expand, 4, 3);
	constexpr int E1 = Slice(Expand, 4, 2);
	constexpr int E2 = Slice(Expand, 4, 1);
	constexpr int E3 = Slice(Expand, 4, 0);
	constexpr int EXPAND_SIZE = E0 * E1 * E2 * E3;
	EType in_all[SafeArrayLength(N_IN)*EXPAND_SIZE*MAX_PTR];
#include "stub/fore1234.h"
#pragma unroll
		for (int i = OFS_IN; i < OFS_IN+N_IN; ++i) {
			const auto &ium = iums[i];
			const int n_ptr = Slice(PtrNum, 4, i);
			const int ci0 = i0*(1-Slice(Opt, 1, DIM*i+0));
			const int ci1 = i1*(1-Slice(Opt, 1, DIM*i+1));
			const int ci2 = i2*(1-Slice(Opt, 1, DIM*i+2));
			const int ci3 = i3*(1-Slice(Opt, 1, DIM*i+3));
			const int ceid = ((ci0*E1+ci1)*E2+ci2)*E3+ci3;
			auto ofs = LUT[ium.eblock_mofs+eid];
			if (Stage == 1) {
				ofs += LUT[ium.accum_step_mofs+asid];
			}
#pragma unroll
			for (int j = 0; j < n_ptr; ++j) {
				in_all[((i-OFS_IN)*MAX_PTR+j)*EXPAND_SIZE+ceid] = shm[i*MAX_PTR+j][ofs];
			}
		}
	}}}}
#include "stub/fore1234.h"
		simple_array<EType, N_IN*MAX_PTR> in;
		simple_array<EType, N_OUT*MAX_PTR> out;
#pragma unroll
		for (int i = OFS_IN; i < OFS_IN+N_IN; ++i) {
			const int n_ptr = Slice(PtrNum, 4, i);
			const int ci0 = i0*(1-Slice(Opt, 1, DIM*i+0));
			const int ci1 = i1*(1-Slice(Opt, 1, DIM*i+1));
			const int ci2 = i2*(1-Slice(Opt, 1, DIM*i+2));
			const int ci3 = i3*(1-Slice(Opt, 1, DIM*i+3));
			const int ceid = ((ci0*E1+ci1)*E2+ci2)*E3+ci3;
			for (int j = 0; j < n_ptr; ++j) {
				in[(i-OFS_IN)*MAX_PTR+j] = in_all[((i-OFS_IN)*MAX_PTR+j)*EXPAND_SIZE+ceid];
			}
		}
		CallStrategy(umi_ctx[eid], uc, in, out, AnInt<Stage>());
		if (mvalid[eid]) {
#pragma unroll
			for (int i = OFS_OUT; i < OFS_OUT+N_OUT; ++i) {
				const int n_ptr = Slice(PtrNum, 4, i+TOTAL_IN);
				const auto &oum = oums[i];
				auto ofs = LUT[oum.eblock_mofs+eid];
				if (Stage == 1) {
					ofs += LUT[oum.accum_block_mofs+abid];
					ofs += LUT[oum.accum_step_mofs+asid];
				}
#pragma unroll
				for (int j = 0; j < n_ptr; ++j) {
					glb[i*MAX_PTR+j][ofs] = out[(i-OFS_OUT)*MAX_PTR+j];
				}
			}
		}
	}}}}
}

template <
	class Umi,
	uint64_t PtrNum, // TODO: Document is required
	uint64_t Expand, // TODO: Document is required
	uint64_t ZeroStrideOpt, // TODO: Document is required
	class EType = typename remove_cv<typename Umi::ElementType>::type,
	class CType = typename remove_cv<typename Umi::UserConstantType>::type
> __global__ void Execute_gpukernel(
	const PDims_packed pdims,
	const ADims_packed adims,
	const int n_exec_thread,
	const int n_load_warp,
	const CType user_const,
	const simple_array<UnrollITranslator_packed<EType>, Umi::TOTAL_IN> ium,
	const simple_array<UnrollOTranslator_packed<EType>, Umi::TOTAL_OUT> oum,
	const Int4 *grid_idx,
	const Int4 *eblock_idx,
	const Int4 *warp_idx,
	const Int4 *thread_idx,
	const uint16_t accum_block_idx,
	const uint16_t accum_step_idx
) {
	constexpr int E0 = Slice(Expand, 4, 3);
	constexpr int E1 = Slice(Expand, 4, 2);
	constexpr int E2 = Slice(Expand, 4, 1);
	constexpr int E3 = Slice(Expand, 4, 0);
	constexpr int EXPAND_SIZE = E0 * E1 * E2 * E3;
	extern __shared__ char cshm[];
	EType *eshm = reinterpret_cast<EType*>(cshm);
	const int BID = blockIdx.x,
	          TID = threadIdx.x,
	          WID = TID/WARP_SIZE,
	          WTID = TID%WARP_SIZE;
	Int4 pidx_base = grid_idx[BID];
	bool mvalid[EXPAND_SIZE];
	Umi umi_ctx[EXPAND_SIZE];
	EType* shm[SafeArrayLength(Umi::TOTAL_IN)*MAX_PTR];
	EType* glb[SafeArrayLength(Umi::TOTAL_OUT)*MAX_PTR];
	constexpr int PRE_I_OFS = 0;
	constexpr int PRE_O_OFS = 0;
	constexpr int LOOP_I_OFS = PRE_I_OFS + Umi::N_PRE_IN;
	constexpr int LOOP_O_OFS = PRE_O_OFS + Umi::N_PRE_OUT;
	constexpr int POST_I_OFS = LOOP_I_OFS + Umi::N_LOOP_IN;
	constexpr int POST_O_OFS = LOOP_O_OFS + Umi::N_LOOP_OUT;
	//////////////////////////////////////////////////
	// ofs and idx stage
	//////////////////////////////////////////////////
	if (TID < n_exec_thread) {
		// Calculate idx
		pidx_base += warp_idx[WID];
		pidx_base += thread_idx[WTID];
		for (int i = 0; i < EXPAND_SIZE; ++i) {
			Int4 pidx = pidx_base;
			pidx += eblock_idx[i];
			mvalid[i] = pidx < pdims.totaldims_;
		}
		// Calculate base
#pragma unroll
		for (int i = 0; i < Umi::TOTAL_IN; ++i) {
			const int n_ptr = Slice(PtrNum, 4, i);
#pragma unroll
			for (int j = 0; j < n_ptr; ++j) {
				shm[i*MAX_PTR+j] =
					eshm +
					ium[i].shm_ofs[j] +
					LUT[ium[i].warp_mofs+WID] +
					LUT[ium[i].thread_mofs+WTID];
			}
		}
#pragma unroll
		for (int i = 0; i < Umi::TOTAL_OUT; ++i) {
			const int n_ptr = Slice(PtrNum, 4, Umi::TOTAL_IN+i);
#pragma unroll
			for (int j = 0; j < n_ptr; ++j) {
				glb[i*MAX_PTR+j] =
					oum[i].ptr[j] +
					oum[i].grid_mofs[BID] +
					LUT[oum[i].warp_mofs+WID] +
					LUT[oum[i].thread_mofs+WTID];
			}
		}
	}

	// pre stage
	FillShm<PtrNum, PRE_I_OFS, Umi::N_PRE_IN, 0>(eshm, BID, 0, TID, n_load_warp, ium);
	CoreExecution<
		Expand, PtrNum, ZeroStrideOpt, 0,
		PRE_I_OFS, PRE_O_OFS,
		Umi::N_PRE_IN, Umi::N_PRE_OUT,
		Umi::TOTAL_IN
	>(glb, shm, ium, oum, 0, 0, mvalid, umi_ctx, user_const);
	if (Umi::N_PRE_IN != 0) {
		__syncthreads();
	}

	// loop stage
	const int agridsize = adims.griddims_.Prod();
	const int ablocksize = adims.blockdims_.Prod();
	for (int abid = 0; abid < agridsize; ++abid) {
		FillShm<PtrNum, LOOP_I_OFS, Umi::N_LOOP_IN, true>(eshm, BID, abid, TID, n_load_warp, ium);
		if (TID < n_exec_thread) {
			const auto abid_ofs = accum_block_idx+4*abid;
			const auto lastofs_ofs = accum_step_idx + (ablocksize-1)*4;
			Int4 last = Int4(
				LUT[abid_ofs  ],
				LUT[abid_ofs+1],
				LUT[abid_ofs+2],
				LUT[abid_ofs+3]
			);
			last += Int4(
				LUT[lastofs_ofs  ],
				LUT[lastofs_ofs+1],
				LUT[lastofs_ofs+2],
				LUT[lastofs_ofs+3]
			);
			const bool accum_all_valid = last < adims.totaldims_;
			if (accum_all_valid) {
				for (int asid = 0; asid < ablocksize; ++asid) {
					CoreExecution<
						Expand, PtrNum, ZeroStrideOpt, 1,
						LOOP_I_OFS, LOOP_O_OFS,
						Umi::N_LOOP_IN, Umi::N_LOOP_OUT,
						Umi::TOTAL_IN
					>(glb, shm, ium, oum, abid, asid, mvalid, umi_ctx, user_const);
				}
			} else {
				for (int asid = 0; asid < ablocksize; ++asid) {
					const auto abid_ofs = accum_block_idx+4*abid;
					const auto asid_ofs = accum_step_idx+4*asid;
					Int4 cur = Int4(
						LUT[abid_ofs  ],
						LUT[abid_ofs+1],
						LUT[abid_ofs+2],
						LUT[abid_ofs+3]
					);
					cur += Int4(
						LUT[asid_ofs  ],
						LUT[asid_ofs+1],
						LUT[asid_ofs+2],
						LUT[asid_ofs+3]
					);
					if (cur < adims.totaldims_) {
						CoreExecution<
							Expand, PtrNum, ZeroStrideOpt, 1,
							LOOP_I_OFS, LOOP_O_OFS,
							Umi::N_LOOP_IN, Umi::N_LOOP_OUT,
							Umi::TOTAL_IN
						>(glb, shm, ium, oum, abid, asid, mvalid, umi_ctx, user_const);
					}
				}
			}
		}
		if (Umi::N_LOOP_IN != 0) {
			__syncthreads();
		}
	}

	// post stage
	FillShm<PtrNum, POST_I_OFS, Umi::N_POST_IN, 2>(eshm, BID, 0, TID, n_load_warp, ium);
	CoreExecution<
		Expand, PtrNum, ZeroStrideOpt, 2,
		POST_I_OFS, POST_O_OFS,
		Umi::N_POST_IN, Umi::N_POST_OUT,
		Umi::TOTAL_IN
	>(glb, shm, ium, oum, 0, 0, mvalid, umi_ctx, user_const);
}

template <class T>
struct PackUtility {
	void Pack(const vector<T> &rhs) {
		ofs.push_back(v.size());
		v.insert(v.end(), rhs.begin(), rhs.end());
	}
	T* Unpack() {
		return ptr+ofs[i++];
	}
	size_t UnpackOffset()
	{
		return ofs[i++];
	}
	const T& operator[](size_t i) { return v[i];}
	void Allocate() {
		const auto S = sizeof(T)*v.size();
		cudaMalloc(&ptr, S);
		cudaMemcpy(ptr, v.data(), S, cudaMemcpyHostToDevice);
	}
	~PackUtility() {
		cudaFree(ptr);
	}
	const vector<T>& get_vector()
	{
		return v;
	}
	T *ptr = nullptr;
	size_t i = 0;
	vector<size_t> ofs;
	vector<T> v;
};

template<uint64_t PtrNum, class Um>
inline int32_t CalculateShm(Um &ium, int &ofs, const int n)
{
	int32_t shm_size = 0;
	for (int i = 0; i < n; ++i) {
		const int32_t s = CeilAlign<int32_t>(ium[ofs+i].shm_boundaries.Sum()+1, WARP_SIZE);
		for (int j = 0; j < Slice(PtrNum, 4, ofs+i); ++j) {
			ium[ofs+i].shm_ofs[j] = shm_size;
			shm_size += s;
		}
	}
	ofs += n;
	return shm_size;
}

#define PrintI(v) {for (auto &x:v) printf("%d\n", x); puts("-----");}
#define PrintI4(v) {for (auto &x:v) x.Print(); puts("-----");}
template <
	class Umi,
	uint64_t PtrNum = 0x1111111111111111llu, // TODO: Document is required
	uint64_t Expand = 0x1111, // TODO: Document is required
	uint64_t ZeroStrideOpt = 0, // TODO: Document is required
	class EType = typename remove_cv<typename Umi::ElementType>::type,
	class CType = typename remove_cv<typename Umi::UserConstantType>::type
> typename enable_if<sizeof(EType) == WORD_SIZE>::type Execute(
	const NDTuple<ParallelDim> &p_dims,
	const NDTuple<AccumulationDim> &a_dims,
	const array<UnrolledI<EType>, Umi::TOTAL_IN> &imems,
	const array<UnrolledO<EType>, Umi::TOTAL_OUT> &omems,
	const CType &c = CType()
) {
	Int4 expand(Slice(Expand, 4, 3), Slice(Expand, 4, 2), Slice(Expand, 4, 1), Slice(Expand, 4, 0));
	PDims_packed p_dims_packed(p_dims, expand);
	ADims_packed a_dims_packed(a_dims);
	simple_array<UnrollITranslator_packed<EType>, Umi::TOTAL_IN> ium;
	simple_array<UnrollOTranslator_packed<EType>, Umi::TOTAL_OUT> oum;
	// tuple of six vector<Int4>
	auto indices = PrecomputeIdx(p_dims_packed, a_dims_packed);
	const size_t S0 = get<0>(indices).size();
	const size_t S1 = get<1>(indices).size();
	const size_t S2 = get<2>(indices).size();
	const size_t S3 = get<3>(indices).size();
	const size_t S4 = get<4>(indices).size();
	const size_t S5 = get<5>(indices).size();
	// These variables are used for offseting in buffer
	PackUtility<Int4> all_Int4;
	PackUtility<Int4> all_Int4_const;
	PackUtility<int> all_int;
	PackUtility<int> all_int_const;
	// Fill the idx table
	all_Int4.Pack(get<0>(indices));
	all_Int4.Pack(get<1>(indices));
	all_Int4.Pack(get<2>(indices));
	all_Int4.Pack(get<3>(indices));
	all_Int4_const.Pack(get<4>(indices));
	all_Int4_const.Pack(get<5>(indices));
	// Fill the translator
	// TODO: the content of the for loops are quite similar, merge them
	for (int i = 0; i < Umi::TOTAL_IN; ++i) {
		ViewDims_packed pmv(imems[i].p_view_dims);
		ViewDims_packed amv(imems[i].a_view_dims);
		PhysicalDims_packed phy(imems[i].physical_dims);
		const Int4 &mpdid = pmv.dim_ids_;
		const Int4 &madid = amv.dim_ids_;
		const Int4 &mpstr = pmv.strides_;
		const Int4 &mastr = amv.strides_;
		const Int4 &mpofs = pmv.ofss_;
		const Int4 &maofs = amv.ofss_;
		// Calculate SHM shape
		Int4 shape(1);
		/*if (true) */ {
			Int4 b = p_dims_packed.eblockdims_;
			b -= 1;
			b *= pmv.strides_;
			shape.ShuffleAccumulate(b, pmv.dim_ids_);
		}
		if (Umi::N_PRE_IN <= i and i < Umi::N_PRE_IN+Umi::N_LOOP_IN) {
			Int4 b = a_dims_packed.blockdims_;
			b -= 1;
			b *= amv.strides_;
			shape.ShuffleAccumulate(b, amv.dim_ids_);
		}
		// User provided SHM padding is ignored in this version
		// const Int4 shape(imems[i].shm_dims);
		Int4 tmp;
		tmp = phy.sizes_;
		ium[i].glb_boundaries = phy.sizes_;
		tmp.CumprodE();
		ium[i].glb_boundaries.CumprodI();
		ium[i].glb_boundaries -= tmp;

		tmp = shape;
		ium[i].shm_boundaries = shape;
		tmp.CumprodE();
		ium[i].shm_boundaries.CumprodI();
		ium[i].shm_boundaries -= tmp;

		ium[i].glb_modes = phy.modes_;
		copy(begin(imems[i].padding), end(imems[i].padding), begin(ium[i].pad));
		// Calculate offset tables
		Int4 gbase(0);
		gbase.ShuffleAccumulate(mpofs, mpdid);
		gbase.ShuffleAccumulate(maofs, madid);
		      vector<Int4   > grid_mofs        = PrecomputeMofs      (get<0>(indices), mpdid, mpstr, gbase);
		const vector<int32_t> eblock_mofs      = PrecomputeMofsLinear(get<1>(indices), mpdid, mpstr, shape);
		const vector<int32_t> warp_mofs        = PrecomputeMofsLinear(get<2>(indices), mpdid, mpstr, shape);
		const vector<int32_t> thread_mofs      = PrecomputeMofsLinear(get<3>(indices), mpdid, mpstr, shape);
		      vector<Int4   > accum_block_mofs = PrecomputeMofs      (get<4>(indices), madid, mastr       );
		const vector<int32_t> accum_step_mofs  = PrecomputeMofsLinear(get<5>(indices), madid, mastr, shape);
		Int4 gstride = phy.sizes_, sstride = shape;
		gstride.CumprodE();
		sstride.CumprodE();
		for (auto &g: grid_mofs) g *= gstride;
		for (auto &g: accum_block_mofs) g *= gstride;
		// Calculate SHM table
		auto smofs = PrecomputeShmofs(shape, gstride, sstride);
		// Fill the struct
		copy(begin(imems[i].ptr), end(imems[i].ptr), begin(ium[i].ptr));
		ium[i].shm_load_num = get<0>(smofs).size();
		ium[i].load_thread_effective = get<1>(smofs).size();
		all_Int4.Pack(grid_mofs);
		all_Int4.Pack(accum_block_mofs);
		all_Int4.Pack(get<0>(smofs));
		all_Int4.Pack(get<1>(smofs));
		all_Int4.Pack(get<2>(smofs));
		all_Int4.Pack(get<3>(smofs));
		all_int_const.Pack(eblock_mofs);
		all_int_const.Pack(warp_mofs);
		all_int_const.Pack(thread_mofs);
		all_int_const.Pack(accum_step_mofs);
	}
	for (int i = 0; i < Umi::TOTAL_OUT; ++i) {
		ViewDims_packed pmv(omems[i].p_view_dims);
		ViewDims_packed amv(omems[i].a_view_dims);
		PhysicalDims_packed phy(omems[i].physical_dims);
		const Int4 &mpdid = pmv.dim_ids_;
		const Int4 &madid = amv.dim_ids_;
		const Int4 &mpstr = pmv.strides_;
		const Int4 &mastr = amv.strides_;
		const Int4 &mpofs = pmv.ofss_;
		const Int4 &maofs = amv.ofss_;
		const Int4 &shape = phy.sizes_;
		// Calculate offset tables
		Int4 gbase(0);
		gbase.ShuffleAccumulate(mpofs, mpdid);
		gbase.ShuffleAccumulate(maofs, madid);
		const vector<int32_t> grid_mofs         = PrecomputeMofsLinear(get<0>(indices), mpdid, mpstr, shape, gbase);
		const vector<int32_t> eblock_mofs       = PrecomputeMofsLinear(get<1>(indices), mpdid, mpstr, shape       );
		const vector<int32_t> warp_mofs         = PrecomputeMofsLinear(get<2>(indices), mpdid, mpstr, shape       );
		const vector<int32_t> thread_mofs       = PrecomputeMofsLinear(get<3>(indices), mpdid, mpstr, shape       );
		const vector<int32_t> accum_block_mofs  = PrecomputeMofsLinear(get<4>(indices), madid, mastr, shape       );
		const vector<int32_t> accum_step_mofs   = PrecomputeMofsLinear(get<5>(indices), madid, mastr, shape       );
		// Fill the struct
		copy(begin(omems[i].ptr), end(omems[i].ptr), begin(oum[i].ptr));
		all_int.Pack(grid_mofs);
		all_int_const.Pack(eblock_mofs);
		all_int_const.Pack(warp_mofs);
		all_int_const.Pack(thread_mofs);
		all_int_const.Pack(accum_block_mofs);
		all_int_const.Pack(accum_step_mofs);
	}
	// Assign the real address/ofs of buffers
	// Constant memory
	uint16_t accum_block_idx_d, accum_step_idx_d;
	vector<int32_t> LUT_temp(CONSTANT_MEMORY_SIZE);
	auto all_int_cv = all_int_const.get_vector();
	auto all_Int4_cv = all_Int4_const.get_vector();
	size_t Int4_base = all_int_cv.size();
	size_t total_const = Int4_base + all_Int4_cv.size()*4;
	assert(total_const <= CONSTANT_MEMORY_SIZE);
	copy(all_int_cv.begin(), all_int_cv.end(), LUT_temp.begin());
	for(size_t i = Int4_base, ii = 0; i < total_const; i += 4, ++ii) {
		LUT_temp[i  ] = all_Int4_cv[ii][0];
		LUT_temp[i+1] = all_Int4_cv[ii][1];
		LUT_temp[i+2] = all_Int4_cv[ii][2];
		LUT_temp[i+3] = all_Int4_cv[ii][3];
	}
	cudaMemcpyToSymbol(LUT, LUT_temp.data(), sizeof(int32_t)*CONSTANT_MEMORY_SIZE);
	accum_block_idx_d = Int4_base + 4*all_Int4_const.UnpackOffset();
	accum_step_idx_d = Int4_base + 4*all_Int4_const.UnpackOffset();
	// Global memory
	Int4 *grid_idx_d, *eblock_idx_d, *warp_idx_d, *thread_idx_d;
	all_Int4.Allocate();
	all_int.Allocate();
	grid_idx_d        = all_Int4.Unpack();
	eblock_idx_d      = all_Int4.Unpack();
	warp_idx_d        = all_Int4.Unpack();
	thread_idx_d      = all_Int4.Unpack();
	for (int i = 0; i < Umi::TOTAL_IN; ++i) {
		ium[i].grid_mofs        = all_Int4.Unpack();
		ium[i].accum_block_mofs = all_Int4.Unpack();
		ium[i].warp_glbshm_mofs    = all_Int4.Unpack();
		ium[i].subwarp_glbshm_mofs = all_Int4.Unpack();
		ium[i].warp_shmshm_mofs    = all_Int4.Unpack();
		ium[i].subwarp_shmshm_mofs = all_Int4.Unpack();
		ium[i].eblock_mofs     = all_int_const.UnpackOffset();
		ium[i].warp_mofs       = all_int_const.UnpackOffset();
		ium[i].thread_mofs     = all_int_const.UnpackOffset();
		ium[i].accum_step_mofs = all_int_const.UnpackOffset();
	}
	for (int i = 0; i < Umi::TOTAL_OUT; ++i) {
		oum[i].grid_mofs        = all_int.Unpack();
		oum[i].eblock_mofs      = all_int_const.UnpackOffset();
		oum[i].warp_mofs        = all_int_const.UnpackOffset();
		oum[i].thread_mofs      = all_int_const.UnpackOffset();
		oum[i].accum_block_mofs = all_int_const.UnpackOffset();
		oum[i].accum_step_mofs  = all_int_const.UnpackOffset();
	}
	// Calculate required shared memory
	int32_t max_shm = 0;
	int ofs = 0;
	max_shm = max(max_shm, CalculateShm<PtrNum>(ium, ofs, Umi::N_PRE_IN));
	max_shm = max(max_shm, CalculateShm<PtrNum>(ium, ofs, Umi::N_LOOP_IN));
	max_shm = max(max_shm, CalculateShm<PtrNum>(ium, ofs, Umi::N_POST_IN));
	const int n_exec_thread = S2*WARP_SIZE;
	// NOTE: we do not use native warp here (default = 64)
	const int n_load_warp = max<int>(S2/MIN_LOAD_THREAD, 1);
	const int n_launch_thread = max(n_load_warp*MIN_LOAD_THREAD, n_exec_thread);
	Execute_gpukernel<Umi, PtrNum, Expand, ZeroStrideOpt><<<S0, n_launch_thread, max_shm*sizeof(EType)>>>(
		p_dims_packed,
		a_dims_packed,
		n_exec_thread,
		n_load_warp,
		c, ium, oum,
		grid_idx_d,
		eblock_idx_d,
		warp_idx_d,
		thread_idx_d,
		accum_block_idx_d,
		accum_step_idx_d
	);
}

} // end using namespace UMI
