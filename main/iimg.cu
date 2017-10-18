#include "UMI_gpu.h"
#include "SyncedMemory.h"
#include <cstdio>
#include <functional>
#include <vector>
#include <numeric>
#include <memory>
using namespace std;
using namespace UMI;

struct Filter3x3 {
	int accum;
	struct Constant {};
	STRATEGY_TYPEDEF_STUB(int, Constant)
	STRATEGY_PRE_STUB(0,0) {accum = 0;}
	STRATEGY_LOOP_STUB(1,1) {OMEM(0,0) = accum += IMEM(0,0);}
	STRATEGY_POST_STUB_
};

int main(int argc, char const* argv[])
{
	const int w = 500;
	const int h = 300;
	MemoryBuffer<int> inb(w*h);
	MemoryBuffer<int> tmpb(w*h);
	MemoryBuffer<int> outb(w*h);
	auto ins = inb.CreateSync(w*h);
	auto tmps = tmpb.CreateSync(w*h);
	auto outs = outb.CreateSync(w*h);
	int *in_cpu = ins.get_cpu_wo();
	fill(in_cpu, in_cpu+w*h, 1);

	const int *in_gpu = ins.get_gpu_ro();
	int *tmp_gpu = tmps.get_gpu_wo();
	int *out_gpu = outs.get_gpu_wo();
	const auto PAD = UMI::PhysicalDim::BoundaryMode::PAD;
	Execute<Filter3x3, 0x11>(
		{{{1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {h,128,32,1}}},
		{{{1,1}, {1,1}, {1,1}, {w,32}}},
		{{
			{
				{{{}, {}, {}, {2,0,1}}},
				{{{}, {}, {}, {3,0,1}}},
				{{{1, PAD}, {1, PAD}, {h, PAD}, {w, PAD}}},
				DEFAULT_SHM,
				{in_gpu},
				{}
			}
		}},
		{{
			{
				{{{}, {}, {}, {3,0,1}}},
				{{{}, {}, {}, {2,0,1}}},
				{{{1, PAD}, {1, PAD}, {w, PAD}, {h, PAD}}},
				{tmp_gpu}
			}
		}},
		{}
	);
	Execute<Filter3x3, 0x11>(
		{{{1,1,1}, {1,1,1}, {1,1,1}, {w,128,32}}},
		{{{1,1}, {1,1}, {1,1}, {h,32}}},
		{{
			{
				{{{}, {}, {}, {2,0,1}}},
				{{{}, {}, {}, {3,0,1}}},
				{{{1, PAD}, {1, PAD}, {w, PAD}, {h, PAD}}},
				{}, // shm
				{tmp_gpu},
				{}
			}
		}},
		{{
			{
				{{{}, {}, {}, {3,0,1}}},
				{{{}, {}, {}, {2,0,1}}},
				{{{1, PAD}, {1, PAD}, {h, PAD}, {w, PAD}}},
				{out_gpu}
			}
		}},
		{}
	);

	const int *ours = outs.get_cpu_ro();
	const int *tmp_cpu = tmps.get_cpu_ro();
	unique_ptr<int[]> tmp(new int[w*h]);
	unique_ptr<int[]> ans(new int[w*h]);
	for (int x = 0, rowsum = 0; x < w; ++x) {
		ans[x] = rowsum += in_cpu[x];
	}
	for (int y = 1; y < h; ++y) {
		int rowsum = 0;
		for (int x = 0; x < w; ++x) {
			rowsum += in_cpu[w*y+x];
			ans[w*y+x] = rowsum + ans[w*(y-1)+x];
		}
	}
	assert(equal(ours, ours+w*h, ans.get()));

	for (int y = 0; y < h; ++y) {
		int rowsum = 0;
		for (int x = 0; x < w; ++x) {
			tmp[w*y+x] = rowsum += in_cpu[w*y+x];
		}
	}
	for (int x = 0; x < w; ++x) {
		int colsum = 0;
		for (int y = 0; y < h; ++y) {
			ans[w*y+x] = colsum += tmp[w*y+x];
		}
	}
	assert(equal(ours, ours+w*h, ans.get()));
	return 0;
}
