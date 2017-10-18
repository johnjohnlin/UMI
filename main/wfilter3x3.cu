#include "UMI_gpu.h"
#include "SyncedMemory.h"
#include <memory>
#include <numeric>
using namespace std;
using namespace UMI;

struct WFilter3x3 {
	int accum;
	struct Constant {int init;};
	STRATEGY_TYPEDEF_STUB(int, Constant)
	STRATEGY_PRE_STUB(0,0) {accum = c.init;}
	STRATEGY_LOOP_STUB(2,0) {accum += IMEM(0,0)*IMEM(1,0);}
	STRATEGY_POST_STUB(0,1) {OMEM(0,0) = accum;}
};

int main(int argc, char const* argv[])
{
	const int w = 300;
	const int h = 250;
	const int init = -5;
	const int padding = 56;
	int weights[9] = {1,2,1,2,4,2,1,2,1};
	MemoryBuffer<int> inb(w*h);
	MemoryBuffer<int> outb(w*h);
	MemoryBuffer<int> wb(9);
	auto ins = inb.CreateSync(w*h);
	auto outs = outb.CreateSync(w*h);
	auto ws = wb.CreateSync(9);
	int *in_cpu = ins.get_cpu_wo();
	iota(in_cpu, in_cpu+w*h, 0);
	int *w_cpu = ws.get_cpu_wo();
	copy(weights, weights+9, w_cpu);

	const int *in_gpu = ins.get_gpu_ro();
	const int *w_gpu = ws.get_gpu_ro();
	int *out_gpu = outs.get_gpu_wo();
	const auto PAD = UMI::PhysicalDim::BoundaryMode::PAD;
	Execute<WFilter3x3, 0x111, 0x1122, 0xc0>(
		{{{1,1,1,1}, {1,1,1,1}, {h,16,4,1}, {w,16,8,1}}},
		{{{1,1}, {1,1}, {3,3}, {3,3}}},
		{{
			{
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{{{}, {}, {2,-1,1},{3,-1,1}}},
				{{{1, PAD}, {1, PAD}, {h, PAD}, {w, PAD}}},
				DEFAULT_SHM,
				{in_gpu},
				{padding} // padding
			},
			{
				{{{}, {}, {}, {}}},
				{{{}, {}, {2,0,1},{3,0,1}}},
				{{{1, PAD}, {1, PAD}, {3, PAD}, {3, PAD}}},
				DEFAULT_SHM,
				{w_gpu},
				{}
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
		{init}
	);

	const int *ours = outs.get_cpu_ro();
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			int sum = init;
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					const int xx = x+dx;
					const int yy = y+dy;
					const int weight = weights[3*(dy+1)+(dx+1)];
					if (xx < 0 or xx >= w or yy < 0 or yy >= h) {
						sum += weight * padding;
					} else {
						sum += weight * in_cpu[w*yy+xx];
					}
				}
			}
			assert(sum == ours[w*y+x]);
		}
	}
	return 0;
}
