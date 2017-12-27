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
	const int b = 8;
	const int sr_ofs = -7;
	const int sr = 15;
	MemoryBuffer<int> frameb(w*h);
	MemoryBuffer<int> sadb((w/b)*(h/b)*sr*sr);
	auto frames = frameb.CreateSync(w*h);
	auto sads = sadb.CreateSync((w/b)*(h/b)*sr*sr);
	int *frame_cpu = frames.get_cpu_wo();
	iota(frame_cpu, frame_cpu+w*h, 0);

	const int *frame_gpu = frames.get_gpu_ro();
	int *sad_gpu = sads.get_gpu_wo();
	const auto WRAP = UMI::PhysicalDim::BoundaryMode::WRAP;
	Execute<MotionEst, 0x111, 0x2211>(
		{{{h/b,1,1,1}, {w/b,1,1,1}, {sr,16,2,1}, {sr,16,16,1}}},
		{{{1,1}, {1,1}, {b,b}, {b,b}}},
		{{
			{
				{{{2,0,b}, {3,0,b}, {2,sr_ofs,1}, {3,sr_ofs,1}}},
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{{{1,WRAP}, {1,WRAP}, {h,WRAP}, {w,WRAP}}},
				DEFAULT_SHM,
				{frame_gpu},
				{}
			},
			{
				{{{2,0,b}, {3,0,b}, {}, {}}},
				{{{}, {}, {2,0,1}, {3,0,1}}},
				{{{1,WRAP}, {1,WRAP}, {h,WRAP}, {w,WRAP}}},
				DEFAULT_SHM,
				{frame_gpu},
				{}
			}
		}},
		{{
			{
				{{{0,0,1}, {1,0,1}, {2,0,1}, {3,0,1}}},
				{},
				{{{h/b,WRAP}, {w/b,WRAP}, {sr,WRAP}, {sr,WRAP}}},
				{sad_gpu}
			}
		}},
		{}
	);

	const int *ours = sads.get_cpu_ro();
	size_t c = 0;
	for (int y = 0; y < h; y += b) {
		for (int x = 0; x < w; x += b) {
			for (int sy = 0; sy < sr; ++sy) {
				for (int sx = 0; sx < sr; ++sx) {
					int sum = 0;
					for (int py = 0; py < b; ++py) {
						for (int px = 0; px < b; ++px) {
							int spy = y+py+sy+sr_ofs;
							int spx = x+px+sx+sr_ofs;
							spy = max(min(spy, h-1), 0);
							spx = max(min(spx, w-1), 0);
							sum += abs(frame_cpu[(y+py)*w+(x+px)] - frame_cpu[spy*w+spx]);
						}
					}
					assert(ours[c] == sum);
					++c;
				}
			}
		}
	}

	return 0;
}
