
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

int models[1000000];
int volatile thread_batch_on[8];

void _do(int * data, int* nelems, int * indices, int ** myindices, int * nelem, int thread_id, int n_batches){
  
  for (int batch = 0; batch < n_batches; batch++) {

    thread_batch_on[thread_id] = batch;    

    /*for (int i = 0; i < 8; i++) {
      cout << thread_batch_on[i] << " ";
    }
    cout << endl;
    std::cout << "THREAD: " << thread_id << " STARTING BATCH " << batch << std::endl;*/

    int waiting_for_other_threads = 1;
    while (waiting_for_other_threads) {
      waiting_for_other_threads = 0;
      for (int ii = 0; ii < 8; ii++) {
	if (thread_batch_on[ii] < batch) {
	  //if (thread_id == 1) std::cout << "I'M ON BATCH " << batch << " WAITING ON " << thread_batch_on[ii] << " OF " << ii << std::endl;
	  waiting_for_other_threads = 1;
	  break;
	}
      }
    }
    
    //std::cout << "THREAD: " << thread_id << " ON BATCH " << batch << std::endl;
    unsigned short p_rand_seed[3];
    p_rand_seed[0] = rand();
    p_rand_seed[1] = rand();
    p_rand_seed[2] = rand();
    
    for(int iid=0;iid<nelem[batch];iid++){
      int eid = myindices[batch][iid];
      //int nelem = nelems[eid];
      int k = nelems[eid];
      int nstart = indices[eid];
      
      float a = 0.0;
      for(int i=0;i<GRAD_COST;i++){
	a += erand48(p_rand_seed);
      }
      
      //std::cout << eid << " " << nelem << " " << nstart << std::endl;
      for(int j=nstart;j<nstart+k;j++){
	//std::cout << data[j] << std::endl;
	models[data[j]] += a;
      }
    }
  }
}


int main(int argc, char ** argv){

  // first, load data
  const int NTHREAD = 8;
  int nelem = 0;
  int ct = 0;
  int value;
  std::vector<int> p_examples;
  std::vector<int> p_nelems;
  std::vector<int> indices;
  std::vector<int> values;

  std::ifstream fin("data/dataaccess_data_multinomialLogisticRegression.txt");
  while(fin >> nelem){
    p_examples.push_back(ct);
    p_nelems.push_back(nelem);
    for(int i=0;i<nelem;i++){
      fin >> value;
      indices.push_back(value);
      values.push_back(1);
      ct ++;
    }
  }

  //First, count number of batches from file
  std::ifstream count_batches_file("data/dataaccess_nthreads8_multinomialLogisticRegression.txt");
  std::string   line;
  int n_batches = 0;
  while (std::getline(count_batches_file, line)) {
    n_batches ++;
  }
  n_batches /= NTHREAD;
  count_batches_file.close();    

  int NELEMS[n_batches][NTHREAD];
  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[j][i] = 0;
    }
  }
  std::ifstream file("data/dataaccess_nthreads8_multinomialLogisticRegression.txt");
  while(std::getline(file, line)){
    std::stringstream   linestream(line);
    std::string         data;

    int batch_id, thread, exampleid;


    // If you have truly tab delimited data use getline() with third parameter.
    // If your data is just white space separated data
    // then the operator >> will do (it reads a space separated word into a string).
    //std::getline(linestream, data, '\t');  // read up-to the first tab (discard tab).

    linestream >> batch_id >> thread;
    //std::cout << epoch << " " << thread << std::endl;
    while(linestream >> exampleid){
      NELEMS[batch_id][thread] ++;
    }
  }

  //numa aware indieces : triple array of form [batch_id][thread][Edge in CC as data point id]
  int* numa_aware_indices[n_batches][NTHREAD];
  for(int ithread=0;ithread<NTHREAD;ithread++){
    numa_run_on_node(ithread / (NTHREAD / 2));
    numa_set_localalloc();
    //std::cout << ithread << "~" << NELEMS[ithread] << std::endl;
    for (int i = 0; i < n_batches; i++) {
      //numa_aware_indices[i][ithread] = new int[NELEMS[i][ithread]];
      numa_aware_indices[i][ithread] = (int *)numa_alloc_onnode(NELEMS[i][ithread] * sizeof(int), ithread % 2);
    }
  }

  for(int i=0;i<NTHREAD;i++){
    for (int j = 0; j < n_batches; j++) {
      NELEMS[j][i] = 0;
    }
  }

  std::vector<int> exampleids;
  for(int j=0;j<p_examples.size();j++){
    exampleids.push_back(j);
  }
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  std::ifstream file2("data/dataaccess_nthreads8_multinomialLogisticRegression.txt");
  while(std::getline(file2, line)){
    std::stringstream   linestream(line);
    std::string         data;

    int batch, thread, exampleid;
    linestream >> batch >> thread;
    while(linestream >> exampleid){
      numa_aware_indices[batch][thread][NELEMS[batch][thread]] = exampleid;
      NELEMS[batch][thread] ++;
    }
  }

  Timer t;
  for(int iepoch=0;iepoch<N_EPOCHS;iepoch++){
    std::vector<std::thread> threads;
    for(int i=0;i<NTHREAD;i++){
      numa_run_on_node(i / (NTHREAD / 2));
      threads.push_back(std::thread(
				    _do, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices[i], NELEMS[i], i, n_batches
				    ));
    }
    for(int i=0;i<threads.size();i++){
      threads[i].join();
    }
  }
  std::cout << t.elapsed() << std::endl;
  return 0;
}


















