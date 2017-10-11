#include "UMI_gpu.h"
#include "SyncedMemory.h"
#include <cstdio>
#include <functional>
#include <vector>
#include <numeric>
using namespace std;
using namespace UMI;

struct Filter3x3 {
	int accum;
	struct Constant {int init, mul;};
	STRATEGY_TYPEDEF_STUB(int, Constant)
	STRATEGY_PRE_STUB(0,0) {accum = c.init;}
	STRATEGY_LOOP_STUB(1,0) {accum += IMEM(0,0);}
	STRATEGY_POST_STUB(0,1) {OMEM(0,0) = accum * c.mul;}
};

int main(int argc, char const* argv[])
{
	const int w = 500;
	const int h = 300;
	const int init = -5;
	const int mul = 2;
	const int padding = 999;
	MemoryBuffer<int> inb(w*h);
	MemoryBuffer<int> outb(w*h);
	auto ins = inb.CreateSync(w*h);
	auto outs = outb.CreateSync(w*h);
	int *in_cpu = ins.get_cpu_wo();
	iota(in_cpu, in_cpu+w*h, 0);

	const int *in_gpu = ins.get_gpu_ro();
	int *out_gpu = outs.get_gpu_wo();
	const auto PAD = UMI::PhysicalDim::BoundaryMode::PAD;
	Execute<Filter3x3, 0x11, 0x1111, 0x1122>(
		{{{1,1,1}, {1,1,1}, {h,16,4}, {w,16,8}}},
		{{{1,1}, {1,1}, {3,3}, {3,3}}},
		{{
			{
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{{{}, {}, {2,-1,1},{3,-1,1}}},
				{{{1, PAD}, {1, PAD}, {h, PAD}, {w, PAD}}},
				DEFAULT_SHM,
				{in_gpu},
				{padding} // padding
			}
		}},
		{{
			{
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{}, // not necessary
				{{{}, {}, {h, PAD}, {w, PAD}}},
				{out_gpu}
			}
		}},
		{init, mul}
	);

	const int *ours = outs.get_cpu_ro();
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			int sum = init;
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					const int xx = x+dx;
					const int yy = y+dy;
					if (xx < 0 or xx >= w or yy < 0 or yy >= h) {
						sum += padding;
					} else {
						sum += in_cpu[w*yy+xx];
					}
				}
			}
			sum *= mul;
			assert(sum == ours[w*y+x]);
		}
	}

	return 0;
}
