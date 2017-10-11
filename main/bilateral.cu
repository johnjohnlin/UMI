#include "UMI_gpu.h"
#include "SyncedMemory.h"
#include <memory>
#include <numeric>
#include <cmath>
using namespace std;
using namespace UMI;

struct Bilateral {
	float wsum, wxsum, center;
	struct Constant {float norm;};
	STRATEGY_TYPEDEF_STUB(float, Constant)
	STRATEGY_PRE_STUB(1,0) {
		wsum = wxsum = 0;
		center = IMEM(0,0);
	}
	STRATEGY_LOOP_STUB(3,0) {
		float n = IMEM(0,0), d = n-center;
		float w = expf(d*d*c.norm)*IMEM(1,0)*IMEM(2,0);
		wsum += w;
		wxsum += w*n;
	}
	STRATEGY_POST_STUB(0,1) {OMEM(0,0) = wxsum / wsum;}
};

int main(int argc, char const* argv[])
{
	const int w = 300;
	const int h = 250;
	const float sigma_1 = 4.0f;
	const float sigma_2 = 10.0f;
	const float norm_1 = -1.0f/(2.0f*sigma_1*sigma_1);
	const float norm_2 = -1.0f/(2.0f*sigma_2*sigma_2);
	const int ofs = sigma_1*3.0f;
	const int k = 2*ofs+1;
	MemoryBuffer<float> inb(w*h);
	MemoryBuffer<float> outb(w*h);
	MemoryBuffer<float> weightb(k);
	auto ins = inb.CreateSync(w*h);
	auto outs = outb.CreateSync(w*h);
	auto weights = weightb.CreateSync(k);
	float *in_cpu = ins.get_cpu_wo();
	iota(in_cpu, in_cpu+w*h, 0);
	float *w_cpu = weights.get_cpu_wo();
	for (int i = 0; i < k; ++i) {
		const int d = i-ofs;
		w_cpu[i] = exp(d*d*norm_1);
	}

	const float *in_gpu = ins.get_gpu_ro();
	const float *w_gpu = weights.get_gpu_ro();
	float *out_gpu = outs.get_gpu_wo();
	const auto PAD = UMI::PhysicalDim::BoundaryMode::PAD;
	Execute<Bilateral, 0x11111>(
		{{{1,1,1}, {1,1,1}, {h,16,2}, {w,16,16}}},
		{{{1,1}, {1,1}, {k,k}, {k,k}}},
		{{
			{
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{{{}, {}, {}, {}}},
				{{{1, PAD}, {1, PAD}, {h, PAD}, {w, PAD}}},
				DEFAULT_SHM,
				{in_gpu},
				{} // padding
			},
			{
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{{{}, {}, {2,-ofs,1}, {3,-ofs,1}}},
				{{{1, PAD}, {1, PAD}, {h, PAD}, {w, PAD}}},
				DEFAULT_SHM,
				{in_gpu},
				{0.0} // padding
			},
			{
				{{{}, {}, {}, {}}},
				{{{}, {}, {3,0,1}, {}}},
				{{{1, PAD}, {1, PAD}, {1, PAD}, {k, PAD}}},
				DEFAULT_SHM,
				{w_gpu},
				{}
			},
			{
				{{{}, {}, {}, {}}},
				{{{}, {}, {}, {3,0,1}}},
				{{{1, PAD}, {1, PAD}, {1, PAD}, {k, PAD}}},
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
		{norm_2}
	);
	puts(cudaGetErrorString(cudaGetLastError()));

	const float *ours = outs.get_cpu_ro();
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			float wsum = 0.0f, wxsum = 0.0f;
			for (int dy = -ofs; dy <= ofs; ++dy) {
				for (int dx = -ofs; dx <= ofs; ++dx) {
					const int xx = x+dx;
					const int yy = y+dy;
					float cur = in_cpu[w*y+x], n = 0.0f;
					if (xx >= 0 and xx < w and yy >= 0 and yy < h) {
						n = in_cpu[w*yy+xx];
					}
					float d = n-cur, w = expf(d*d*norm_2)*w_cpu[dy+ofs]*w_cpu[dx+ofs];
					wsum += w;
					wxsum += w*n;
				}
			}
			float a = wxsum/wsum;
			float b = ours[w*y+x];
			assert(a<1.001f*b and a>0.999f*b);
		}
	}
	return 0;
}
