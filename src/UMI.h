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
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>
#ifdef __CUDA_ARCH__
#  define DHFunc __device__ __host__
#  define DFunc __device__
#else
#  define DHFunc
#  define DFunc
#endif

namespace UMI {

// TODO: "dim_ids" is confusing along with "dims"
// and the "dims" is replaced by "shapes" or "sizes" in the newer parts of the code.
// Must be unified to shapes.

// Constants
template<class Int> inline constexpr bool IsPowOf2(const Int i) { return i > 0 and (i&(i-1)) == 0; }
static constexpr int MAX_PTR = 4;
static constexpr int DIM = 4;
static constexpr int MAX_WARP = 32;
static constexpr int WARP_SIZE = 32;
static constexpr int WORD_SIZE = 4;
static constexpr int MAX_BLOCK_SIZE = MAX_WARP*WARP_SIZE;
static constexpr int LOAD_WARP = 2;
static constexpr int MIN_LOAD_THREAD = WARP_SIZE*LOAD_WARP;
static_assert(IsPowOf2(WARP_SIZE), "Warp size must be positive and power of 2");
using namespace std;

// Helper class/templates/functions
constexpr DHFunc int SafeArrayLength(int i) {return i>=1? i: 1;}
template<class T, size_t N>
struct simple_array {
	DHFunc T& operator[](size_t i) {return data[i];}
	DHFunc const T& operator[](size_t i) const {return data[i];}
	DHFunc simple_array() {}
	simple_array(const array<T, N> &a) {
		copy(a.begin(), a.end(), begin(data));
	}
private:
	T data[SafeArrayLength(N)];
};
template <class T> using identity_t = typename enable_if<true, T>::type;
template <class T> using NDTuple = array<T, DIM>;
template <class Int> inline constexpr Int CeilDiv(const Int a, const Int b) {return (a-1)/b+1;}
template <class Int> inline constexpr Int CeilDivBit(const Int a, const Int b) {return ((a-1)>>b)+1;}
template <class Int> inline constexpr Int CeilAlign(const Int a, const Int b) {return CeilDiv(a,b)*b;}
template <class Int> inline constexpr Int CeilAlignBit(const Int a, const Int b) {return CeilDivBit(a,b)<<b;}
template <class Int> inline constexpr Int CeilAlignBitExact(const Int a, const Int b) {return (CeilDivBit(a,b)|Int(1))<<b;}
constexpr DHFunc uint64_t Slice(const uint64_t i, int b, int nth) {
	return i<<(64-(nth+1)*b)>>(64-b);
}

struct AccumulationDim {
	int32_t total_size, block_size;
};

struct ParallelDim {
	int32_t total_size, block_size, warp_size, warp_shuf;
};

struct ViewDim {
	int32_t physical_dim_id, ofs, stride;
};

struct PhysicalDim {
	enum BoundaryMode: int32_t {PAD, WRAP};
	int32_t size;
	BoundaryMode mode;
};

template<typename T>
struct UnrolledI {
	NDTuple<ViewDim> p_view_dims;
	NDTuple<ViewDim> a_view_dims;
	NDTuple<PhysicalDim> physical_dims;
	NDTuple<int32_t> shm_dims;
	const T* ptr[MAX_PTR];
	const T padding[MAX_PTR];
};

template<typename T>
struct UnrolledO {
	NDTuple<ViewDim> p_view_dims;
	NDTuple<ViewDim> a_view_dims;
	NDTuple<PhysicalDim> physical_dims;
	T* ptr[MAX_PTR];
};

#define STRATEGY_TYPEDEF_STUB(e, c)\
	typedef e ElementType;\
	typedef c UserConstantType;\
	template <int s> using A = UMI::simple_array<ElementType, s>;
#define STRATEGY_PRE_STUB(ni, no)\
	static constexpr int N_PRE_IN = ni;\
	static constexpr int N_PRE_OUT = no;\
	DHFunc void Pre(const UserConstantType &c, const A<N_PRE_IN*MAX_PTR> &in, A<N_PRE_OUT*MAX_PTR> &out)
#define STRATEGY_LOOP_STUB(ni, no)\
	static constexpr int N_LOOP_IN = ni;\
	static constexpr int N_LOOP_OUT = no;\
	DHFunc void Loop(const UserConstantType &c, const A<N_LOOP_IN*MAX_PTR> &in, A<N_LOOP_OUT*MAX_PTR> &out)
#define STRATEGY_POST_STUB(ni, no)\
	static constexpr int N_POST_IN = ni;\
	static constexpr int N_POST_OUT = no;\
	static constexpr int TOTAL_IN = N_PRE_IN + N_LOOP_IN + N_POST_IN;\
	static constexpr int TOTAL_OUT = N_PRE_OUT + N_LOOP_OUT + N_POST_OUT;\
	static constexpr int TOTAL = TOTAL_IN + TOTAL_OUT;\
	DHFunc void Post(const UserConstantType &c, const A<N_POST_IN*MAX_PTR> &in, A<N_POST_OUT*MAX_PTR> &out)
#define STRATEGY_POST_PRE_ STRATEGY_PRE_STUB(0, 0){}
#define STRATEGY_LOOP_STUB_ STRATEGY_LOOP_STUB(0, 0){}
#define STRATEGY_POST_STUB_ STRATEGY_POST_STUB(0, 0){}
#define IMEM(i, j) in[(i)*MAX_PTR+(j)]
#define OMEM(i, j) out[(i)*MAX_PTR+(j)]
#define DEFAULT_SHM {-1,-1,-1,-1}

// packed int32_t
static_assert(DIM == 4, "Currently only DIM=4 is supported.");
struct Int4 {
	alignas(16) int32_t data[4];
	DHFunc inline void Set(int v = 0) {
#pragma unroll 4
		for (int i = 0; i < 4; ++i) {
			data[i] = v;
		}
	}
	DHFunc Int4(int32_t v) {
		Set(v);
	}
	Int4(const NDTuple<int32_t> &a) {
		copy(begin(a), end(a), begin(data));
	}
	DHFunc Int4(int32_t a, int32_t b, int32_t c, int32_t d):
		data{a, b, c, d} {}
	DHFunc Int4() = default;
	DHFunc Int4(const Int4 &rhs) = default;
	DHFunc Int4& operator=(const Int4 &) = default;
	DHFunc inline int32_t& operator[](const int i) {
		return data[i];
	}
	DHFunc inline const int32_t& operator[](const int i) const {
		return data[i];
	}
	// Since we use pragma preprocessor in these functions
	// So the include workaround is necessary
#define OP +=
#include "stub/Int4_binary.h"
#undef OP
#define OP -=
#include "stub/Int4_binary.h"
#undef OP
#define OP *=
#include "stub/Int4_binary.h"
#undef OP
#define OP ==
#include "stub/Int4_cmp.h"
#undef OP
#define OP !=
#include "stub/Int4_cmp.h"
#undef OP
// NOTE: these operators means a OP b for all elements
// Be careful that a > b != not a <= b
#define OP <
#include "stub/Int4_cmp.h"
#undef OP
#define OP <=
#include "stub/Int4_cmp.h"
#undef OP
#define OP >
#include "stub/Int4_cmp.h"
#undef OP
#define OP >=
#include "stub/Int4_cmp.h"
#undef OP
	inline int32_t Unfold(const Int4 &dims) const {
		int32_t unfolded = 0;
#pragma unroll 4
		for (int i = 0; i < 4; ++i) {
			unfolded = unfolded * dims.data[i] + data[i];
		}
		return unfolded;
	}
	DHFunc inline int ClampedSum(const Int4 &boundaries, const Int4 &modes, bool &outside) const {
		int32_t s = 0;
		outside = false;
#pragma unroll 4
		for (int i = 0; i < 4; ++i) {
			int32_t v = data[i];
			int32_t boundary = boundaries.data[i];
			int32_t mode = modes.data[i];
			int32_t v_clamp = max(min(v, boundary), 0);
			s += v_clamp;
			outside |= (v != v_clamp) and (mode == PhysicalDim::BoundaryMode::PAD);
		}
		return s;
	}
	DHFunc inline int ClampedSum(const Int4 &boundaries, bool &outside) const {
		int32_t s = 0;
		outside = false;
#pragma unroll 4
		for (int i = 0; i < 4; ++i) {
			int32_t v = data[i];
			int32_t boundary = boundaries.data[i];
			int32_t v_clamp = max(min(v, boundary), 0);
			s += v_clamp;
			outside |= (v != v_clamp);
		}
		return s;
	}
	inline void IncreaseOne(const Int4 &dims) {
		for (int i = 3; i >= 0; --i) {
			++data[i];
			if (data[i] == dims.data[i]) {
				data[i] = 0;
			} else {
				break;
			}
		}
	}
	inline void ShuffleAccumulate(const Int4 &val, const Int4 &dim_id) {
		for (int i = 0; i < 4; ++i) {
			int d = dim_id[i];
			int v = val[i];
			data[d] += v;
		}
	}
	DHFunc inline int32_t Sum() const {
		int32_t s = 0;
#pragma unroll 4
		for (int i = 0; i < 4; ++i) {
			s += data[i];
		}
		return s;
	}
	DHFunc inline int32_t Prod() const {
		int32_t s = 1;
#pragma unroll 4
		for (int i = 0; i < 4; ++i) {
			s *= data[i];
		}
		return s;
	}
	inline void CumprodE() {
		int32_t s = 1;
		for (int i = 3; i >= 0; --i) {
			const int32_t d = data[i];
			data[i] = s;
			s *= d;
		}
	}
	inline void CumprodI() {
		int32_t s = 1;
		for (int i = 3; i >= 0; --i) {
			const int32_t d = data[i];
			s *= d;
			data[i] = s;
		}
	}
	void Print() const;
};

struct PDims_packed {
	Int4 totaldims_;
	Int4 griddims_;
	Int4 blockdims_;
	Int4 warpdims_;
	Int4 warpshufs_;
	Int4 expand_;
	// derived
	Int4 large_warpdims_;
	Int4 blockdims_in_warp_;
	Int4 eblockdims_;
	PDims_packed(const NDTuple<ParallelDim> &dims, const Int4 &expand);
};

struct ADims_packed {
	Int4 totaldims_;
	Int4 griddims_;
	Int4 blockdims_;
	ADims_packed(const NDTuple<AccumulationDim> &dims);
};

struct ViewDims_packed {
	Int4 dim_ids_;
	Int4 ofss_;
	Int4 strides_;
	DHFunc ViewDims_packed() {}
	ViewDims_packed(const NDTuple<ViewDim> &dims);
};

struct PhysicalDims_packed {
	Int4 sizes_;
	Int4 modes_;
	DHFunc PhysicalDims_packed() {}
	PhysicalDims_packed(const NDTuple<PhysicalDim> &dims);
};

template<class EType>
struct UnrollITranslator_packed {
	Int4 glb_boundaries;
	Int4 shm_boundaries;
	Int4 glb_modes;
	const EType *ptr[MAX_PTR];
	EType pad[MAX_PTR];
	int32_t load_thread_effective;
	int32_t shm_ofs[MAX_PTR];
	int32_t shm_load_num;
	// pointer type: global memory
	// int type: offset type in packed constant memory
	Int4 *grid_mofs;
	Int4 *accum_block_mofs;
	Int4 *warp_glbshm_mofs;
	Int4 *subwarp_glbshm_mofs;
	Int4 *warp_shmshm_mofs;
	Int4 *subwarp_shmshm_mofs;
	uint16_t eblock_mofs;
	uint16_t warp_mofs;
	uint16_t thread_mofs;
	uint16_t accum_step_mofs;
};

template<class EType>
struct UnrollOTranslator_packed {
	EType *ptr[MAX_PTR];
	// pointer type: global memory
	// int type: offset type in packed constant memory
	int32_t *grid_mofs;
	uint16_t eblock_mofs;
	uint16_t warp_mofs;
	uint16_t thread_mofs;
	uint16_t accum_block_mofs;
	uint16_t accum_step_mofs;
};

tuple<
	vector<Int4> /* grid_ofs        */,
	vector<Int4> /* eblock_ofs      */,
	vector<Int4> /* warp_ofs        */,
	vector<Int4> /* thread_ofs      */,
	vector<Int4> /* accum_block_ofs */,
	vector<Int4> /* accum_step_ofs  */
> PrecomputeIdx(const PDims_packed &p_dims, const ADims_packed &a_dims);

vector<Int4> PrecomputeMofs(
	const vector<Int4> &idxs,
	const Int4 &dims,
	const Int4 &strides,
	const Int4 &base_mofs = Int4(0)
);

vector<int> PrecomputeMofsLinear(
	const vector<Int4> &idxs,
	const Int4 &dims,
	const Int4 &strides,
	const Int4 &shapes,
	const Int4 &base_mofs = Int4(0)
);

tuple<
	vector<Int4> /* glb_load_warp_mofs   */,
	vector<Int4> /* glb_load_thread_mofs */,
	vector<Int4> /* shm_load_warp_mofs   */,
	vector<Int4> /* shm_load_thread_mofs */
> PrecomputeShmofs(const Int4 &shm_shapes, const Int4 &glb_strides, const Int4 &shm_strides);

} // end using namespace UMI
