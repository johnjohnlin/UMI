#include "UMI_gpu.h"
#include "SyncedMemory.h"
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>
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
	// 7x7+2x2s
	const int iw = 80;
	const int ih = 64;
	const int f = 7;
	const int ofs = -f/2;
	const int s = 2;
	const int ich = 40;
	const int och = 60;
	const int ow = iw/s;
	const int oh = ih/s;
	const int batchsize = 32;
	const int isize = iw*ih*ich*batchsize;
	const int wsize = och*ich*f*f;
	const int osize = ow*oh*och*batchsize;
	MemoryBuffer<int> inb(isize);
	MemoryBuffer<int> weightsb(wsize);
	MemoryBuffer<int> outb(osize);
	auto ins = inb.CreateSync(isize);
	auto weightss = weightsb.CreateSync(wsize);
	auto outs = outb.CreateSync(osize);
	default_random_engine gen;
	uniform_int_distribution<int> dist(0,10);
	auto rgen = bind(dist, gen);
	int *in_cpu = ins.get_cpu_wo();
	int *weights_cpu = weightss.get_cpu_wo();
	generate_n(in_cpu, isize, ref(rgen));
	generate_n(weights_cpu, wsize, ref(rgen));

	const int *weights_gpu = weightss.get_gpu_ro();
	const int *in_gpu = ins.get_gpu_ro();
	int *out_gpu = outs.get_gpu_wo();
	const auto PAD = UMI::PhysicalDim::BoundaryMode::PAD;
	Execute<Gemm>(
		{{{batchsize,1,1}, {och,4,1}, {oh,8,4}, {ow,8,8}}},
		{{{1,1}, {ich,6}, {f,f}, {f,f}}},
		{{
			{
				{{{0,0,1}, {}, {2,0,s}, {3,0,s}}},
				{{{}, {1,0,1}, {2,ofs,1}, {3,ofs,1}}},
				{{{batchsize, PAD}, {ich, PAD}, {ih, PAD}, {iw, PAD}}},
				DEFAULT_SHM,
				{in_gpu},
				{}
			},
			{
				{{{}, {0,0,1}, {}, {}}},
				{{{}, {1,0,1}, {2,0,1}, {3,0,1}}},
				{{{och, PAD}, {ich, PAD}, {f, PAD}, {f, PAD}}},
				DEFAULT_SHM,
				{weights_gpu},
				{}
			}
		}},
		{{
			{
				{{{0,0,1},{1,0,1},{2,0,1},{3,0,1}}},
				{},
				{{{batchsize, PAD}, {och, PAD}, {oh, PAD}, {ow, PAD}}},
				{out_gpu}
			}
		}}
	);

	const int *ours = outs.get_cpu_ro();
	for (int i = 0; i < batchsize; ++i) {
		size_t c = 0;
		for (int oc = 0; oc < och; ++oc) {
			for (int iy = 0; iy < ih; iy += s) {
				for (int ix = 0; ix < iw; ix += s) {
					int sum = 0;
					for (int ic = 0; ic < ich; ++ic) {
						for (int dy = 0; dy < f; ++dy) {
							for (int dx = 0; dx < f; ++dx) {
								const int xx = ix+dx+ofs;
								const int yy = iy+dy+ofs;
								const int weight = weights_cpu[oc*ich*f*f+ic*f*f+dy*f+dx];
								if (0 <= xx and xx < iw and 0 <= yy and yy < ih) {
									sum += weight * in_cpu[i*iw*ih*ich+ic*iw*ih+yy*iw+xx];
								}
							}
						}
					}
					assert(ours[i*och*ow*oh+c] == sum);
					++c;
				}
			}
		}
	}
	return 0;
}
