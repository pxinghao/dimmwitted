
#include <iostream>
#include "src/util.h"
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <algorithm> 

#include <unistd.h>

int models[1000000];
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

    //std::cout << eid << " " << nelem << " " << nstart << std::endl;
    for(int j=nstart;j<nstart+nelem;j++){
      //std::cout << data[j] << std::endl;
      models[data[j]] += a;
    }

  }
}


int main(int argc, char ** argv){

  // first, load data
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

  const int NTHREAD = 8;
  int NELEMS[NTHREAD];
  int NELEMS_HOGWILD[NTHREAD];
  for(int i=0;i<NTHREAD;i++){
    NELEMS[i] = 0;
    NELEMS_HOGWILD[i] = p_examples.size() / NTHREAD;
  }
  std::ifstream file("data/dataaccess_nthreads8_multinomialLogisticRegression.txt");
  std::string   line;
  while(std::getline(file, line)){
    std::stringstream   linestream(line);
    std::string         data;

    int epoch, thread, exampleid;


    // If you have truly tab delimited data use getline() with third parameter.
    // If your data is just white space separated data
    // then the operator >> will do (it reads a space separated word into a string).
    //std::getline(linestream, data, '\t');  // read up-to the first tab (discard tab).

    linestream >> epoch >> thread;
    //std::cout << epoch << " " << thread << std::endl;
    while(linestream >> exampleid){
      NELEMS[thread] ++;
    }
  }

  int* numa_aware_indices[NTHREAD];
  int* numa_aware_indices_hogwild[NTHREAD];
  const int NELEM_HOGWILD = p_examples.size() / NTHREAD;
  for(int ithread=0;ithread<NTHREAD;ithread++){
    numa_run_on_node(ithread / (NTHREAD / 2));
    numa_set_localalloc();
    //std::cout << ithread << "~" << NELEMS[ithread] << std::endl;
    numa_aware_indices[ithread] = new int[NELEMS[ithread]];
    numa_aware_indices_hogwild[ithread] = new int[NELEM_HOGWILD];
  }

  for(int i=0;i<NTHREAD;i++){
    NELEMS[i] = 0;
  }

  std::vector<int> exampleids;
  for(int j=0;j<p_examples.size();j++){
    exampleids.push_back(j);
  }
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle (exampleids.begin(), exampleids.end(), std::default_random_engine(seed));
  for(int j=0;j<p_examples.size();j++){
    numa_aware_indices_hogwild[j%NTHREAD][NELEMS[j%NTHREAD]] = exampleids[j];
    //std::cout << exampleids[j] << std::endl;
    NELEMS[j%NTHREAD] ++;
  }
  //std::cout << "A" << std::endl;

  for(int i=0;i<NTHREAD;i++){
    NELEMS[i] = 0;
  }
  std::ifstream file2("data/dataaccess_nthreads8_multinomialLogisticRegression.txt");
  while(std::getline(file2, line)){
    std::stringstream   linestream(line);
    std::string         data;

    int epoch, thread, exampleid;
    linestream >> epoch >> thread;
    while(linestream >> exampleid){
      numa_aware_indices[thread][NELEMS[thread]] = exampleid;
      NELEMS[thread] ++;
    }
  }

  /*
  for(int i=0;i<NTHREAD;i++){
    for(int j=0;j<NELEMS[i];j++){
      std::cout << i << " " << numa_aware_indices[i][j] << std::endl; 
    }
  }
  */

  Timer t;
  for(int iepoch=0;iepoch<N_EPOCHS;iepoch++){

    std::vector<std::thread> threads;
    for(int i=0;i<NTHREAD;i++){
      numa_run_on_node(i / (NTHREAD / 2));
      // The following two strategies are one for Cyclades and one for Hogwild!
      threads.push_back(std::thread(
				    _do, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices[i], NELEMS[i], i
				    //_do, &indices[0], &p_nelems[0], &p_examples[0], numa_aware_indices_hogwild[i], NELEMS_HOGWILD[i], i
      ));
    }
    for (int i = 0; i <NTHREAD; i++)
      threads[i].join();
  }
  std::cout << t.elapsed() << std::endl;

  return 0;
}


















