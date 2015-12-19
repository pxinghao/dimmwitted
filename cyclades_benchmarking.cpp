#include <iostream>
#include "src/util.h"
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <algorithm> 
#include <math.h>

#include <unistd.h>

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

#define NTHREAD 8
#define MODEL_SIZE 1000000
#define DATA_FILE "data/dataaccess_data_multinomialLogisticRegression.txt"
#define DATA_ACCESS_FILE "data/dataaccess_nthreads8_multinomialLogisticRegression.txt"

int models[MODEL_SIZE];
int volatile thread_batch_on[NTHREAD];

void _do(int * data, int* nelems, int * indices, int * myindices, int nelem, int thread_id){
  
  unsigned short p_rand_seed[3];
  p_rand_seed[0] = rand();
  p_rand_seed[1] = rand();
  p_rand_seed[2] = rand();


  for(int iid=0;iid<nelem;iid++){
    int eid = myindices[iid];
    int nelem = nelems[eid];
    int nstart = indices[eid];

    float a = 0.0;
    for(int i=0;i<GRAD_COST;i++){
      a += erand48(p_rand_seed);
    }

    for(int j=nstart;j<nstart+nelem;j++){
      models[data[j]] += a;
    }

  }
}

void _do_cyclades(int * data, int* nelems, int * indices, int ** myindices, int * nelem, int thread_id, int n_batches){
  
  for (int batch = 0; batch < n_batches; batch++) {

    //Wait for all threads to be on the same batch
    thread_batch_on[thread_id] = batch;    
    int waiting_for_other_threads = 1;
    while (waiting_for_other_threads) {
      waiting_for_other_threads = 0;
      for (int ii = 0; ii < 8; ii++) {
	if (thread_batch_on[ii] < batch) {
	  waiting_for_other_threads = 1;
	  break;
	}
      }
    }
    
    unsigned short p_rand_seed[3];
    p_rand_seed[0] = rand();
    p_rand_seed[1] = rand();
    p_rand_seed[2] = rand();
    
    for(int iid=0;iid<nelem[batch];iid++){
      int eid = myindices[batch][iid];
      int k = nelems[eid];
      int nstart = indices[eid];
      
      float a = 0.0;
      for(int i=0;i<GRAD_COST;i++){
	a += erand48(p_rand_seed);
      }
      
      for(int j=nstart;j<nstart+k;j++){
	models[data[j]] += a;
      }
    }
  }
}

void load_data(string file, vector<int> &p_examples, vector<int> &p_nelems, vector<int> &indices) {
  int ct = 0;
  int value = 0;
  int nelem = 0;
  ifstream fin(file);
  while(fin >> nelem){
    p_examples.push_back(ct);
    p_nelems.push_back(nelem);
    for(int i=0;i<nelem;i++){
      fin >> value;
      indices.push_back(value);
      ct ++;
    }
  }
  fin.close();
}

int count_batches_in_file(string file) {
  ifstream count_batches_file(file);
  string line;
  int n_batches = 0;
  while (getline(count_batches_file, line)) {
    n_batches++;
  }
  n_batches /= NTHREAD;
  count_batches_file.close();
  return n_batches;
}

void cyclades_no_sync_or_hogwild(int should_cyc_no_sync) {
  //Load data
  vector<int> p_examples;
  vector<int> p_nelems;
  vector<int> indices;
  vector<int> values;
  load_data(DATA_FILE, p_examples, p_nelems, indices);

  //NELEMS for both cyclades and hogwild
  int NELEMS[NTHREAD];
  int NELEMS_HOGWILD[NTHREAD];
  for(int i=0;i<NTHREAD;i++){
    NELEMS[i] = 0;
    NELEMS_HOGWILD[i] = p_examples.size() / NTHREAD;
  }

  //Count number of elements in cyclades batch without sync
  ifstream file(DATA_ACCESS_FILE);
  string   line;
  while(getline(file, line)){
    stringstream linestream(line);
    string data;
    int epoch, thread, exampleid;

    linestream >> epoch >> thread;
    while(linestream >> exampleid){
      NELEMS[thread] ++;
    }
  }

  //Data for both howild and cyclades w/o sync
  int* numa_aware_indices[NTHREAD];
  int* numa_aware_indices_hogwild[NTHREAD];
  const int NELEM_HOGWILD = p_examples.size() / NTHREAD;
  for(int ithread=0;ithread<NTHREAD;ithread++){
    numa_run_on_node(ithread / (NTHREAD / 2));
    numa_set_localalloc();
    numa_aware_indices[ithread] = new int[NELEMS[ithread]];
    numa_aware_indices_hogwild[ithread] = new int[NELEM_HOGWILD];
  }

  //Reset NELEMS to keep track of element counts
  for(int i=0;i<NTHREAD;i++){
    NELEMS[i] = 0;
  }

  //Hogwild data shuffling
  vector<int> exampleids;
  for(int j=0;j<p_examples.size();j++){
    exampleids.push_back(j);
  }
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  shuffle (exampleids.begin(), exampleids.end(), default_random_engine(seed));
  for(int j=0;j<p_examples.size();j++){
    numa_aware_indices_hogwild[j%NTHREAD][NELEMS[j%NTHREAD]] = exampleids[j];
    NELEMS[j%NTHREAD] ++;
  }

  //Reset NELEMS for element count
  for(int i=0;i<NTHREAD;i++){
    NELEMS[i] = 0;
  }
  
  //Reset fp
  file.clear();
  file.seekg(0);
  
  while(getline(file, line)){
    stringstream linestream(line);
    string data;

    int epoch, thread, exampleid;
    linestream >> epoch >> thread;
    while(linestream >> exampleid){
      numa_aware_indices[thread][NELEMS[thread]] = exampleid;
      NELEMS[thread] ++;
    }
  }

  //Do computation
  Timer t;
  for(int iepoch=0;iepoch<N_EPOCHS;iepoch++){

    vector<thread> threads;
    for(int i=0;i<NTHREAD;i++){
      numa_run_on_node(i / (NTHREAD / 2));
      if (should_cyc_no_sync) {
	threads.push_back(thread(_do, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices[i], NELEMS[i], i));
      }
      else {
	threads.push_back(thread(_do, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices_hogwild[i], NELEMS_HOGWILD[i], i));
      }
    }
    for (int i = 0; i <NTHREAD; i++)
      threads[i].join();
  }
  cout << t.elapsed() << endl;
}

void cyclades_benchmark() {
  //Load data
  vector<int> p_examples;
  vector<int> p_nelems;
  vector<int> indices;
  load_data(DATA_FILE, p_examples, p_nelems, indices);
  
  //First, count number of batches from file
  int n_batches = 0;
  n_batches = count_batches_in_file(DATA_ACCESS_FILE);

  //NELEMS will count # of elements in cyclades batches
  int NELEMS[n_batches][NTHREAD];

  //Clear NELEMS
  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[j][i] = 0;
    }
  }

  //Count number of elements in cyclades batches
  string line;
  ifstream file(DATA_ACCESS_FILE);
  while(getline(file, line)){
    stringstream linestream(line);
    string data;
    int batch_id, thread, exampleid;

    linestream >> batch_id >> thread;
    while(linestream >> exampleid){
      NELEMS[batch_id][thread]++;
    }
  }
  
  //Allocate space on numa nodes
  //numa aware indieces : triple array of form [batch_id][thread][Edge in CC as data point id]
  int* numa_aware_indices[n_batches][NTHREAD];
  for(int ithread=0;ithread<NTHREAD;ithread++){
    numa_run_on_node(ithread / (NTHREAD / 2));
    numa_set_localalloc();
    for (int i = 0; i < n_batches; i++) {
      numa_aware_indices[i][ithread] = (int *)numa_alloc_onnode(NELEMS[i][ithread] * sizeof(int), ithread % 2);
    }
  }

  //Rewind fp
  file.clear();
  file.seekg(0);
  
  //Clear NELEMS to keep track of batch count
  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[j][i] = 0;
    }
  }

  //Read in actual data to numa_aware_indeices
  while(getline(file, line)){
    stringstream linestream(line);
    string data;

    int batch, thread, exampleid;
    linestream >> batch >> thread;
    while(linestream >> exampleid){
      numa_aware_indices[batch][thread][NELEMS[batch][thread]] = exampleid;
      NELEMS[batch][thread]++;
    }
  }
  file.close();

  //Do the computation
  Timer t;
  for(int iepoch=0;iepoch<N_EPOCHS;iepoch++){
    vector<thread> threads;
    for(int i=0;i<NTHREAD;i++){
      numa_run_on_node(i / (NTHREAD / 2));
      threads.push_back(thread(_do_cyclades, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices[i], NELEMS[i], i, n_batches));
    }
    for(int i=0;i<threads.size();i++){
      threads[i].join();
    }
  }
  cout << t.elapsed() << endl;
}

int main(int argc, char ** argv){
  if (CYCLADES) {
    cyclades_benchmark();
  }
  if (CYC_NO_SYNC) {
    cyclades_no_sync_or_hogwild(1);
  }
  if (HOGWILD) {
    cyclades_no_sync_or_hogwild(0);
  }
  
  return 0;
}
