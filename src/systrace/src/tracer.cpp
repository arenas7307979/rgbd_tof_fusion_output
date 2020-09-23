#include "tracer.h"
namespace slam_ros
{
#if ENABLE_TRACE
static Tracer gTracer;
int Tracer::sTraceFD = -1;
#endif
}