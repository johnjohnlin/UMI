#include "UMI_gpu.h"
#include "SyncedMemory.h"
#include <algorithm>
#include <numeric>
using namespace std;
using namespace UMI;

struct MotionEst {
	int sad;
	struct Constant {};
	STRATEGY_TYPEDEF_STUB(int, Constant)
	STRATEGY_PRE_STUB(0,0) {sad = 0;}
	STRATEGY_LOOP_STUB(2,0) {sad += abs(IMEM(0,0) - IMEM(1,0));}
	STRATEGY_POST_STUB(0,1) {OMEM(0,0) = sad;}
};

int main(int argc, char const* argv[])
{
	const int w = 1920;
	const int h = 1080;
	const int k = 7;
	const int sr = 15;
	const int k_ofs = k/2;
	const int sr_ofs = sr/2;
	MemoryBuffer<int> frameb(w*h);
	MemoryBuffer<int> cvb(w*h*sr);
	auto frames = frameb.CreateSync(w*h);
	auto cvs = cvb.CreateSync(w*h*sr);
	int *frame_cpu = frames.get_cpu_wo();
	iota(frame_cpu, frame_cpu+w*h, 0);

	const int *frame_gpu = frames.get_gpu_ro();
	int *cv_gpu = cvs.get_gpu_wo();
	const auto WRAP = UMI::PhysicalDim::BoundaryMode::WRAP;
	Execute<MotionEst, 0x111>(
		{{{1,1,1,1}, {h,4,1,1}, {w,8,2,1}, {sr,16,16,1}}},
		{{{1,1}, {1,1}, {1,1}, {k,k}}},
		{{
			{
				{{{}, {2,0,1}, {3,0,1}, {3,-sr_ofs,1}}},
				{{{}, {}, {}, {3,-k_ofs,1}}},
				{{{1,WRAP}, {1,WRAP}, {h,WRAP}, {w,WRAP}}},
				DEFAULT_SHM,
				{frame_gpu},
				{}
			},
			{
				{{{}, {2,0,1}, {3,0,1}, {}}},
				{{{}, {}, {}, {3,-k_ofs,1}}},
				{{{1,WRAP}, {1,WRAP}, {h,WRAP}, {w,WRAP}}},
				DEFAULT_SHM,
				{frame_gpu},
				{}
			}
		}},
		{{
			{
				{{{}, {1,0,1}, {2,0,1}, {3,0,1}}},
				{},
				{{{1,WRAP}, {h,WRAP}, {w,WRAP}, {sr,WRAP}}},
				{cv_gpu}
			}
		}},
		{}
	);

	const int *ours = cvs.get_cpu_ro();
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int sx = -sr_ofs; sx <= sr_ofs; ++sx) {
				int sad = 0;
				for (int k = -k_ofs; k <= k_ofs; ++k) {
					int refx = max(min(x+k, w-1), 0);
					int searchx = max(min(x+sx+k, w-1), 0);
					sad += abs(frame_cpu[y*w+refx] - frame_cpu[y*w+searchx]);
				}
				assert(*(ours++) == sad);
			}
		}
	}

	return 0;
}
