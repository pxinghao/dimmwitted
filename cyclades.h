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

#define GAMMA .0000001
#define MODEL_SIZE 100000
#define MAX_THREADS 256

double volatile models[MODEL_SIZE];
int volatile thread_batch_on[MAX_THREADS]; //max of 256 threads

void _do_cyclades_gradient(double **sparse_mat, double *target, int *data, int* nelems, int * indices, int ** myindices, int * nelem, int thread_id, int n_batches, int NTHREAD){
  
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

void apply_cyclades(double **sparse_mat, double *target_values, vector<int> &p_examples, vector<int> &p_nelems, vector<int> &indices, int **NELEMS, int ***numa_aware_indices,  int NTHREAD, int N_NUMA_NODES, int n_batches, int N_EPOCHS) {

  for(int iepoch = 0; iepoch < N_EPOCHS; iepoch++){
    vector<thread> threads;
    for(int i = 0 ; i < NTHREAD; i++) {
      numa_run_on_node(i % N_NUMA_NODES);
      threads.push_back(thread(_do_cyclades_gradient, sparse_mat, target_values, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices[i], NELEMS[i], i, n_batches, NTHREAD));
    }
    for(int i = 0; i < threads.size(); i++){
      threads[i].join();
      thread_batch_on[i] = 0;
    } 
    //cout << "LOSS: " << compute_loss(sparse_mat, target_values, &indices[0], &p_examples[0], p_nelems) << endl;
  }
}
