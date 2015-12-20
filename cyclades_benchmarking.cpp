#include <iostream>
#include "src/util.h"
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <algorithm> 
#include <math.h>
#include <time.h>
#include <set>
#include <unistd.h>
#include "cyclades.h"

using namespace std;

#ifndef HOGWILD
#define HOGWILD 0
#endif

#ifndef CYCLADES
#define CYCLADES 0
#endif

#ifndef CYC_NO_SYNC
#define CYC_NO_SYNC 0
#endif

#ifndef MODEL_DUP
#define MODEL_DUP 0
#endif

int main(int argc, char ** argv){
  srand(0);
  for (int i = 0; i < MODEL_SIZE; i++) models[i] = 0;
  for (int i = 0; i < N_NUMA_NODES; i++) representative_thread_on[i] = 0;

  if (MODEL_DUP) {
    cyclades_benchmark_model_dup();
  }
  if (CYCLADES) {
    cyclades_benchmark();
  }
  if (CYC_NO_SYNC) {
    cyclades_no_sync_or_hogwild_benchmark(1);
  }
  if (HOGWILD) {
    cyclades_no_sync_or_hogwild_benchmark(0);
  }
  
  return 0;
}
