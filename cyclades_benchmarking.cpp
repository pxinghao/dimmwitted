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

#define VALUE_RANGE 10
#define GAMMA .00001
#define NTHREAD 8
#define N_NUMA_NODES 2
#define MODEL_SIZE 1000000
#define DATA_FILE "data/dataaccess_data_multinomialLogisticRegression.txt"
#define DATA_ACCESS_FILE "data/dataaccess_nthreads8_multinomialLogisticRegression.txt"

double volatile models[MODEL_SIZE];
int volatile thread_batch_on[NTHREAD];
int volatile representative_thread_on[N_NUMA_NODES]; 
int * model_part[N_NUMA_NODES];

void _do_simulation(int * data, int* nelems, int * indices, int * myindices, int nelem, int thread_id){
  
  unsigned short p_rand_seed[3];
  p_rand_seed[0] = rand();
  p_rand_seed[1] = rand();
  p_rand_seed[2] = rand();

  for(int iid=0;iid<nelem;iid++){
    int eid = myindices[iid];
    int nelem = nelems[eid];
    int nstart = indices[eid];

    double a = 0.0;
    for(int i=0;i<GRAD_COST;i++){
      a += erand48(p_rand_seed);
    }

    for(int j=nstart;j<nstart+nelem;j++){
      models[data[j]] += a;
    }
  }
}

void _do_gradient(double **sparse_mat, double *target, int * data, int* nelems, int * indices, int * myindices, int nelem, int thread_id){
  
  unsigned short p_rand_seed[3];
  p_rand_seed[0] = rand();
  p_rand_seed[1] = rand();
  p_rand_seed[2] = rand();

  for(int iid=0;iid<nelem;iid++){
    int eid = myindices[iid];
    int dim = nelems[eid];
    int nstart = indices[eid];

    //Compute gradient
    double gradient = 0;
    for (int i = 0; i < dim; i++) {
      gradient += sparse_mat[eid][i] * models[data[nstart + i]];
    }
    gradient -= target[eid];
    
    //Apply gradient
    for(int i = nstart; i < nstart+dim; i++){
      models[data[i]] = models[data[i]] - GAMMA * sparse_mat[eid][i-nstart] * gradient;
    }
  }
}

void _do_cyclades_gradient_model_rep(double **sparse_mat, double *target, int *data, int* nelems, int * indices, int ** myindices, int * nelem, int thread_id, int n_batches, int numa_node, int is_representative, int *partition){
  
  for (int batch = 0; batch < n_batches; batch++) {

    //Wait for all threads to be on the same batch
    thread_batch_on[thread_id] = batch;    
    int waiting_for_other_threads = 1;
    while (waiting_for_other_threads) {
      waiting_for_other_threads = 0;
      for (int ii = 0; ii < NTHREAD; ii++) {
	if (thread_batch_on[ii] < batch) {
	  waiting_for_other_threads = 1;
	  break;
	}
      }
    }

    //If a representative thread, update model partition with overall view of model
    if (is_representative) {
      //for (int i = 0; i < 100; i++) 
      //model_part[numa_node][i] = models[i];
      representative_thread_on[numa_node] = batch;
    }

    //Wait for representative threads to be on the same batch
    int waiting_for_representative_thread = 1;
    while (waiting_for_representative_thread) {
      waiting_for_representative_thread = representative_thread_on[numa_node] < batch;
    }

     
    for(int iid = 0; iid < nelem[batch]; iid++){
      int eid = myindices[batch][iid];
      int dim = nelems[eid];
      int nstart = indices[eid];
      
      //Compute gradient
      double gradient = 0;
      for (int i = 0; i < dim; i++) {
	gradient += sparse_mat[eid][i] * model_part[numa_node][data[nstart + i]];
	//gradient += sparse_mat[eid][i] * models[data[nstart + i]];
      }
      gradient -= target[eid];
      
      //Apply gradient
      for(int i = nstart; i < nstart+dim; i++){
	model_part[numa_node][data[i]] = model_part[numa_node][data[i]] - GAMMA * sparse_mat[eid][i-nstart] * gradient;
      }
	//models[data[i]] = models[data[i]] - GAMMA * sparse_mat[eid][i-nstart] * gradient;
    }
    //for (int i = 0; i < 100; i++) {
    //models[i] = model_part[numa_node][i];
    //}
  }
}

void _do_cyclades_gradient(double **sparse_mat, double *target, int *data, int* nelems, int * indices, int ** myindices, int * nelem, int thread_id, int n_batches){
  
  for (int batch = 0; batch < n_batches; batch++) {

    //Wait for all threads to be on the same batch
    thread_batch_on[thread_id] = batch;    
    int waiting_for_other_threads = 1;
    while (waiting_for_other_threads) {
      waiting_for_other_threads = 0;
      for (int ii = 0; ii < NTHREAD; ii++) {
	if (thread_batch_on[ii] < batch) {
	  waiting_for_other_threads = 1;
	  break;
	}
      }
    }
    
    for(int iid = 0; iid < nelem[batch]; iid++){
      int eid = myindices[batch][iid];
      int dim = nelems[eid];
      int nstart = indices[eid];
      
      //Compute gradient
      double gradient = 0;
      for (int i = 0; i < dim; i++) {
	gradient += sparse_mat[eid][i] * models[data[nstart + i]];
      }
      gradient -= target[eid];
      
      //Apply gradient
      for(int i = nstart; i < nstart+dim; i++){
	models[data[i]] = models[data[i]] - GAMMA * sparse_mat[eid][i-nstart] * gradient;
      }
    }
  }
}

void _do_cyclades_simulation(int *data, int* nelems, int * indices, int ** myindices, int * nelem, int thread_id, int n_batches){
  
  for (int batch = 0; batch < n_batches; batch++) {

    //Wait for all threads to be on the same batch
    thread_batch_on[thread_id] = batch;    
    int waiting_for_other_threads = 1;
    while (waiting_for_other_threads) {
      waiting_for_other_threads = 0;
      for (int ii = 0; ii < NTHREAD; ii++) {
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
      
      double a = 0.0;
      for(int i=0;i<GRAD_COST;i++){
	a += erand48(p_rand_seed);
      }
      
      for(int j=nstart;j<nstart+k;j++){
	models[data[j]] += a;
      }
    }
  }
}

double compute_loss(double **sparse_mat, double *target_dat, int *data, int *indices, vector<int> &p_nelems) {

  double loss = 0;
  for (int i = 0; i < p_nelems.size(); i++) {
    int nstart = indices[i];
    
    //Calculate prediction
    double prediction = 0;
    for (int j = 0; j < p_nelems[i]; j++) {
      prediction += sparse_mat[i][j] * models[data[nstart+j]];
    }

    loss += (prediction - target_dat[i]) * (prediction - target_dat[i]);
  }
  return loss / (double)p_nelems.size();
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

double ** generate_random_data_for_access_pattern(vector<int> &indices, vector<int> &p_nelems) {
  double **random_sparse_dat = (double **)calloc(p_nelems.size(), sizeof(double *));
  if (random_sparse_dat == NULL) {
    cout << "FAILED TO ALLOCATE SPARSE DATA" << endl;
    exit(0);
  }
  int indices_track = 0;
  for (int data_point_id = 0; data_point_id < p_nelems.size(); data_point_id++) {
    random_sparse_dat[data_point_id] = (double *)calloc(p_nelems[data_point_id], sizeof(double));
    if (random_sparse_dat[data_point_id] == NULL) {
      cout << "FAILED TO ALLOCATE SPARSE DATA VALUES" << endl;
      exit(0);
    }
    for (int j = 0; j < p_nelems[data_point_id]; j++) {
      random_sparse_dat[data_point_id][j] = rand() % VALUE_RANGE;
    }
  }
  return random_sparse_dat;
}

double * generate_random_target_values(int size) {
  double * target_values = (double *)malloc(sizeof(double) * size);
  for (int i = 0; i < size; i++)
    target_values[i] = rand() % VALUE_RANGE;
  return target_values;
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

void cyclades_no_sync_or_hogwild_benchmark(int should_cyc_no_sync) {
  //Load data
  vector<int> p_examples;
  vector<int> p_nelems;
  vector<int> indices;
  vector<int> values;
  load_data(DATA_FILE, p_examples, p_nelems, indices);

  //Generate random data based on access pattern
  double **random_sparse_data = generate_random_data_for_access_pattern(indices, p_nelems);
  double * target_values = generate_random_target_values(p_nelems.size());

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
    if (NTHREAD == 1) {
      numa_run_on_node(0);
    }
    else {
      numa_run_on_node(ithread / (NTHREAD / N_NUMA_NODES));
    }
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
      if (NTHREAD == 1) {
	numa_run_on_node(0);
      }
      else {
	numa_run_on_node(i / (NTHREAD / N_NUMA_NODES));
      }
      if (should_cyc_no_sync) {
	//threads.push_back(thread(_do_simulation, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices[i], NELEMS[i], i));       
	threads.push_back(thread(_do_gradient, random_sparse_data, target_values, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices[i], NELEMS[i], i));       
      }
      else {
	//threads.push_back(thread(_do_simulation, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices_hogwild[i], NELEMS_HOGWILD[i], i));
	threads.push_back(thread(_do_gradient, random_sparse_data, target_values, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices_hogwild[i], NELEMS_HOGWILD[i], i));
      }
    }
    for (int i = 0; i <NTHREAD; i++)
      threads[i].join();
  }

  cout << t.elapsed() << endl;
  cout << compute_loss(random_sparse_data, target_values, &indices[0], &p_examples[0], p_nelems) << endl;

  //Free data
  for (int i = 0; i < p_nelems.size(); i++) 
    free(random_sparse_data[i]);
  free(random_sparse_data);
  free(target_values);
}

void cyclades_benchmark() {
  
  //Load data access pattern
  vector<int> p_examples;
  vector<int> p_nelems;
  vector<int> indices;
  load_data(DATA_FILE, p_examples, p_nelems, indices);

  //Generate random data based on access pattern
  double **random_sparse_data = generate_random_data_for_access_pattern(indices, p_nelems);
  double * target_values = generate_random_target_values(p_nelems.size());
  
  //First, count number of batches from file
  int n_batches = 0;
  n_batches = count_batches_in_file(DATA_ACCESS_FILE);

  //NELEMS will count # of elements in cyclades batches
  int NELEMS[NTHREAD][n_batches];

  //Clear NELEMS
  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[i][j] = 0;
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
      NELEMS[thread][batch_id]++;
    }
  }
  
  //Allocate space on numa nodes
  //numa aware indices : triple array of form [thread][batch_id][edge]
  int* numa_aware_indices[NTHREAD][n_batches];
  for(int ithread=0;ithread<NTHREAD;ithread++){      
    if (NTHREAD == 1) {
      numa_run_on_node(0);
    }
    else {
      numa_run_on_node(ithread / (NTHREAD / N_NUMA_NODES));
    }
    numa_set_localalloc();
    for (int i = 0; i < n_batches; i++) {
      //numa_aware_indices[ithread][i] = (int *)numa_alloc_onnode(NELEMS[i][ithread] * sizeof(int), ithread / (NTHREAD / N_NUMA_NODES));
      numa_aware_indices[ithread][i] = new int[NELEMS[ithread][i] * sizeof(int)];
    }
  }

  //Rewind fp
  file.clear();
  file.seekg(0);
  
  //Clear NELEMS to keep track of batch count
  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[i][j] = 0;
    }
  }

  //Read in actual data to numa_aware_indeices
  while(getline(file, line)){
    stringstream linestream(line);
    string data;
    int batch, thread, exampleid;
    linestream >> batch >> thread;

    while(linestream >> exampleid){
      numa_aware_indices[thread][batch][NELEMS[thread][batch]] = exampleid;
      NELEMS[thread][batch]++;      
    }
  }
  file.close();

  //Do the computation
  Timer t;
  for(int iepoch = 0; iepoch < N_EPOCHS; iepoch++){
    vector<thread> threads;
    for(int i = 0 ; i < NTHREAD; i++) {
      if (NTHREAD == 1) {
	numa_run_on_node(0);
      }
      else {
	numa_run_on_node(i / (NTHREAD / N_NUMA_NODES));
      }
      threads.push_back(thread(_do_cyclades_gradient, random_sparse_data, target_values, &indices[0], &p_nelems[0], &p_examples[0], (int **)numa_aware_indices[i], (int *)NELEMS[i], i, n_batches));
    }
    for(int i = 0; i < threads.size(); i++){
      threads[i].join();
      thread_batch_on[i] = 0;
    }    
  }
  cout << t.elapsed() << endl;
  cout << compute_loss(random_sparse_data, target_values, &indices[0], &p_examples[0], p_nelems) << endl;

  //Free data
  for (int i = 0; i < p_nelems.size(); i++) 
    free(random_sparse_data[i]);
  free(random_sparse_data);
  free(target_values);
}

void cyclades_benchmark_model_dup() {
  
  //Load data access pattern
  vector<int> p_examples;
  vector<int> p_nelems;
  vector<int> indices;
  load_data(DATA_FILE, p_examples, p_nelems, indices);

  //Generate random data based on access pattern
  double **random_sparse_data = generate_random_data_for_access_pattern(indices, p_nelems);
  double * target_values = generate_random_target_values(p_nelems.size());
  
  //First, count number of batches from file
  int n_batches = 0;
  n_batches = count_batches_in_file(DATA_ACCESS_FILE);

  //NELEMS will count # of elements in cyclades batches
  int NELEMS[NTHREAD][n_batches];

  //Clear NELEMS
  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[i][j] = 0;
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
      NELEMS[thread][batch_id]++;
    }
  }
  
  //Allocate space on numa nodes
  //numa aware indices : triple array of form [thread][batch_id][edge]
  int* numa_aware_indices[NTHREAD][n_batches];
  for(int ithread=0;ithread<NTHREAD;ithread++){      
    if (NTHREAD == 1) {
      numa_run_on_node(0);
    }
    else {
      numa_run_on_node(ithread / (NTHREAD / N_NUMA_NODES));
    }
    numa_set_localalloc();
    for (int i = 0; i < n_batches; i++) {
      //numa_aware_indices[ithread][i] = (int *)numa_alloc_onnode(NELEMS[i][ithread] * sizeof(int), ithread / (NTHREAD / N_NUMA_NODES));
      numa_aware_indices[ithread][i] = new int[NELEMS[ithread][i] * sizeof(int)];
    }
  }

  //Rewind fp
  file.clear();
  file.seekg(0);
  
  //Clear NELEMS to keep track of batch count
  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[i][j] = 0;
    }
  }

  //Read in actual data to numa_aware_indeices
  while(getline(file, line)){
    stringstream linestream(line);
    string data;
    int batch, thread, exampleid;
    linestream >> batch >> thread;

    while(linestream >> exampleid){
      numa_aware_indices[thread][batch][NELEMS[thread][batch]] = exampleid;
      NELEMS[thread][batch]++;  
    }
  }
  file.close();

  //Prepare for model duplication
  for (int i = 0; i < N_NUMA_NODES; i++) {
    int node_run_on = NTHREAD == 1 ? 0 : i / (NTHREAD / N_NUMA_NODES);
    numa_run_on_node(node_run_on);
    numa_set_localalloc();
    model_part[i] = (int *)numa_alloc_onnode(MODEL_SIZE * sizeof(int), node_run_on);
  }

  //Do the computation
  Timer t;
  for(int iepoch = 0; iepoch < N_EPOCHS; iepoch++){
    vector<thread> threads;
    for(int i = 0 ; i < NTHREAD; i++) {
      int node_run_on = NTHREAD == 1 ? 0 : i / (NTHREAD / N_NUMA_NODES);
      numa_run_on_node(node_run_on);
      threads.push_back(thread(_do_cyclades_gradient_model_rep, random_sparse_data, target_values, &indices[0], &p_nelems[0], &p_examples[0], (int **)numa_aware_indices[i], (int *)NELEMS[i], i, n_batches, node_run_on, i % (NTHREAD / N_NUMA_NODES) == 0, (int *)model_part[node_run_on]));
    }
    for(int i = 0; i < threads.size(); i++){
      threads[i].join();
      thread_batch_on[i] = 0;
    }    
  }
  cout << t.elapsed() << endl;
  cout << compute_loss(random_sparse_data, target_values, &indices[0], &p_examples[0], p_nelems) << endl;

  //Free data
  for (int i = 0; i < p_nelems.size(); i++) 
    free(random_sparse_data[i]);
  free(random_sparse_data);
  free(target_values);
  //for (int i = 0; i < N_NUMA_NODES; i++) 
  //free(model_part[i]);
}

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
