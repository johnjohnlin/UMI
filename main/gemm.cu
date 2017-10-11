#include "UMI_gpu.h"
#include "SyncedMemory.h"
#include <algorithm>
#include <numeric>
using namespace std;
using namespace UMI;

struct Gemm {
	int accum;
	struct Constant {};
	STRATEGY_TYPEDEF_STUB(int, Constant)
	STRATEGY_PRE_STUB(0,0) {accum = 0;}
	STRATEGY_LOOP_STUB(2,0) {accum += IMEM(0,0) * IMEM(1,0);}
	STRATEGY_POST_STUB(0,1) {OMEM(0,0) = accum;}
};

int main(int argc, char const* argv[])
{
	const int m = 100;
	const int k = 100;
	const int n = 200;
	MemoryBuffer<int> Ab(m*k);
	MemoryBuffer<int> Bb(k*n);
	MemoryBuffer<int> Cb(m*n);
	auto As = Ab.CreateSync(m*k);
	auto Bs = Bb.CreateSync(k*n);
	auto Cs = Cb.CreateSync(m*n);
	int *A_cpu = As.get_cpu_wo();
	int *B_cpu = Bs.get_cpu_wo();
	iota(A_cpu, A_cpu+m*k, 0);
	iota(B_cpu, B_cpu+k*n, 0);

	const int *A_gpu = As.get_gpu_ro();
	const int *B_gpu = Bs.get_gpu_ro();
	int *C_gpu = Cs.get_gpu_wo();
	const auto PAD = UMI::PhysicalDim::BoundaryMode::PAD;
	Execute<Gemm, 0x111, 0x1124, 0x48>(
		{{{1,1,1}, {1,1,1}, {m,16,2}, {n,16,16}}},
		{{{1,1}, {1,1}, {1,1}, {k,16}}},
		{{
			{
				{{{}, {}, {2,0,1}, {}}},
				{{{}, {}, {}, {3,0,1}}},
				{{{1, PAD}, {1, PAD}, {m, PAD}, {k, PAD}}},
				DEFAULT_SHM,
				{A_gpu},
				{}
			},
			{
				{{{}, {}, {}, {3,0,1}}},
				{{{}, {}, {}, {2,0,1}}},
				{{{1, PAD}, {1, PAD}, {k, PAD}, {n, PAD}}},
				DEFAULT_SHM,
				{B_gpu},
				{}
			}
		}},
		{{
			{
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{}, // not necessary
				{{{}, {}, {m, PAD}, {n, PAD}}},
				{C_gpu}
			}
		}},
		{}
	);

	const int *ours = Cs.get_cpu_ro();
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			int sum = 0;
			for (int kk = 0; kk < k; ++kk) {
				sum += A_cpu[y*k+kk] * B_cpu[kk*n+x];
			}
			assert(sum == ours[y*n+x]);
		}
	}
	return 0;
}
