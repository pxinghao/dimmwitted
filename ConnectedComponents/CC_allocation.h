#include <iostream>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <thread>
#include "../src/util.h"
#include <fstream>
#include <string>
#include <algorithm> 
#include <math.h>
#include <time.h>
#include <set>

#include <unistd.h>


using namespace std;

void CC_batch(double ** data_points, int base_point, int num_data_points, int dimension,  vector<vector<int> > &CC_output) {
  //Create bpartite graph. Datapoint edges are from 0 .... num_data_points-1, coordinates from num_data_points ... num_data_points + dimension
  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> g (num_data_points + dimension);
  for (int i = 0; i < num_data_points; i++) {
    for (int j = 0; j < dimension; j++) {
      if (data_points[i][j] != 0) {
	boost::add_edge(i, j+num_data_points, g);
      }
    }
  }
  //Compute CC
  vector<int> components(num_data_points + dimension);
  int num_total_components = boost::connected_components(g, &components[0]);
  
  map<int, int> groups;
  int remapped_group_id = 0;
  for (int i = 0; i < num_data_points; i++) {
    if (groups.find(components[i]) == groups.end()) {
      groups[components[i]] = remapped_group_id++;
    }
  }

  //Aggregate mapping into array of sets of CC's
  CC_output.resize(remapped_group_id);
  for (int i = 0; i < num_data_points; i++) {
    CC_output[groups[components[i]]].push_back(base_point + i);
  }
}

void CC_allocate(double **data_points, int num_data_points, int dimension, int num_numa_nodes, 
		 int ***numa_aware_indices, int **NELEMS, int BATCH_SIZE, int NTHREAD) {
  vector<thread> threads;

  int num_batches = (int)ceil(num_data_points / (float)BATCH_SIZE);
  vector<vector<int> > CCs[num_batches];

  //Parallelize batches
  int batch_num = 0;
  for (int i = 0; i < num_data_points; i += BATCH_SIZE, batch_num++) {
    int num_elements = min(num_data_points, i + BATCH_SIZE) - i;
    //threads.push_back(thread(CC_batch, &data_points[i], i, num_elements, dimension, ref(CCs[batch_num])));
    CC_batch(&data_points[i], i, num_elements, dimension, CCs[batch_num]);
  }
  
  for (int i = 0; i < num_data_points; i++) {
    for (int j = 0; j < dimension; j++) 
      data_points[i][j] = 0;
  }

  //Join threads
  for (int i = 0; i < threads.size(); i++) {
    threads[i].join();
  }

  //Fill out num_numa nodes and NELEMS
  int thread_chosen = 0;
  for (int k = 0; k < num_batches; k++) {
    for (int i = 0; i < CCs[k].size(); i++) {
      int assign_to_thread = thread_chosen++ % NTHREAD;
      //int assign_to_thread = i % 2;
      int numa_node_to_alloc_on = assign_to_thread % num_numa_nodes;
      //numa_run_on_node(numa_node_to_alloc_on);
      //numa_set_localalloc();
      NELEMS[assign_to_thread][k] = CCs[k][i].size();
      numa_aware_indices[assign_to_thread][k] = (int *)numa_alloc_onnode(CCs[k][i].size() * sizeof(int), numa_node_to_alloc_on);

      for (int j = 0; j < CCs[k][i].size(); j++) {
	numa_aware_indices[assign_to_thread][k][j] = CCs[k][i][j];
      }
    }
  }
  
  cout << "CC DONE..." << endl;
}
