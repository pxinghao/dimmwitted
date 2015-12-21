#include <iostream>
#include "src/util.h"
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <algorithm> 
#include <math.h>
#include <time.h>
#include <map>
#include <unistd.h>
#include "ConnectedComponents/CC_allocation.h"
#include <boost/graph/connected_components.hpp>
#include <boost/graph/adjacency_list.hpp>

#define N_USERS 944
#define N_MOVIES 1683

#define N_NUMA_NODES 2
#define N_EPOCHS 10
#define BATCH_SIZE 2000
#define NTHREAD 4
#define GAMMA .0001

#define RLENGTH 20

using namespace std;

typedef tuple<int, int, double> DataPoint;

int volatile thread_batch_on[NTHREAD];

//Initialize v and h matrix models
//double ** v_model, **rmodel;
double volatile v_model[N_USERS][RLENGTH];
double volatile u_model[RLENGTH][N_MOVIES];

double compute_loss(vector<DataPoint> &p) {
  double loss = 0;
  for (int i = 0; i < p.size(); i++) {
    int x = get<0>(p[i]),  y = get<1>(p[i]);
    double r = get<2>(p[i]), s = 0, dp = 0;
    for (int i = 0; i < RLENGTH; i++) {
      dp += v_model[x][i] * u_model[i][y];
    }
    double diff = r - dp;
    loss += diff * diff;
  }
  return loss / (double)p.size();
}

void do_cyclades_gradient_descent(vector<int *> &access_pattern, vector<int> &access_length, vector<DataPoint> &points, int thread_id) {
  for (int batch = 0; batch < access_pattern.size(); batch++) {
    
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
    
    //For every data point in the connected component
    for (int i = 0; i < access_length[batch]; i++) {
      //Compute gradient
      DataPoint p = points[access_pattern[batch][i]];
      int x = get<0>(p), y = get<1>(p);
      double r = get<2>(p);
      double gradient = 0;
      for (int j = 0; j < RLENGTH; j++) {
	gradient += v_model[x][j] * u_model[j][y];
      }
      gradient -= r;

      //Perform updates
      for (int j = 0; j < RLENGTH; j++) {
	double new_v = v_model[x][j] - GAMMA * gradient;
	double new_u = u_model[j][y] - GAMMA * gradient;
	v_model[x][j] = new_v;
	u_model[j][y] = new_u;
      }
    }
  }
}

void distribute_ccs(map<int, vector<int> > &ccs, vector<vector<int *> > &access_pattern, vector<vector<int> > &access_length, int batchnum) {
  int total_size_needed[NTHREAD];
  int count = 0;

  for (int i = 0; i < NTHREAD; i++) total_size_needed[i] = 0;

  //Count total size needed for each access pattern
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    vector<int> cc = it->second;
    int chosen_thread = count % NTHREAD;
    total_size_needed[chosen_thread] += cc.size();
  }
    
  //Allocate memory
  int index_count[NTHREAD];
  for (int i = 0; i < NTHREAD; i++) {
    int numa_node = i % N_NUMA_NODES;
    
    access_pattern[i][batchnum] = (int *)numa_alloc_onnode(total_size_needed[i] * sizeof(int), numa_node);
    access_length[i][batchnum] = total_size_needed[i];
    index_count[i] = 0;
  }
  
  //Copy memory over
  count = 0;  
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    vector<int> cc = it->second;
    int chosen_thread = count % NTHREAD;
    for (int i = 0; i < cc.size(); i++) {
      access_pattern[chosen_thread][batchnum][index_count[chosen_thread]+i] = cc[i];
    }
    index_count[chosen_thread] += cc.size();
  }
}

map<int, vector<int> > compute_CC(vector<DataPoint> &points, int start, int end) {
  //Numeric ordering:
  //Note: points.size() to mean end-start
  //data points - [0 ... points.size()]
  //user vars - [points.size() ... points.size() + N_USERS]
  //movie vars - [points.size() + N_USERS .... end]
  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> g (end-start + N_MOVIES + N_USERS);
  for (int i = 0; i < end-start; i++) {
    boost::add_edge(i, end-start+get<0>(points[i+start]), g);
    boost::add_edge(i, end-start+N_USERS+get<1>(points[i+start]), g);
  }
  
  //CC
  vector<int> components(end-start + N_MOVIES + N_USERS);
  int num_total_components = boost::connected_components(g, &components[0]);

  map<int, vector<int> > CCs;
  //set<int> cids;
  for (int i = 0; i < end-start; i++) {
    //cids.insert(components[i]);
    CCs[components[i]].push_back(i + start);
  }
  //cout << cids.size() << endl;

  return CCs;
}

vector<DataPoint> get_movielens_data() {
  vector<DataPoint> datapoints;

  ifstream in("ml-100k/u.data");
  string s;

  while (getline(in, s)) {
    stringstream str_stream(s);
    double userid, movieid, rating;
    str_stream >> userid >> movieid >> rating;
    datapoints.push_back(DataPoint(userid, movieid, rating));
  }
  return datapoints;
}

void cyclades_movielens_completion() {
  
  //Read data nonzero data points
  vector<DataPoint> points = get_movielens_data();

  
  //Access pattern generation
  int n_batches = (int)ceil((points.size() / (double)BATCH_SIZE));

  //Access pattern of form [thread][batch][point_ids]
  vector<vector<int *> > access_pattern(NTHREAD);
  //Access length (number of elements in corresponding cc)
  vector<vector<int > > access_length(NTHREAD);

  for (int i = 0; i < NTHREAD; i++) {
    access_pattern[i].resize(n_batches);
    access_length[i].resize(n_batches);
  }

  //CC Generation
  for (int i = 0; i < n_batches; i++) {
    
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());

    ///Compute connected components of data points
    map<int, vector<int> > cc = compute_CC(points, start, end);

    //Distribute connected components across threads
    distribute_ccs(cc, access_pattern, access_length, i);
  }

  /*for (int i = 0; i < NTHREAD; i++) {
    cout << "THREAD " << i << endl;
    for (int j = 0; j < n_batches; j++) {
      cout << "BATCH " << j << endl;
      cout << access_length[i][j] << " : ";
      for (int k = 0; k < min(10, access_length[i][j]); k++) {
	cout << access_pattern[i][j][k] << " ";
      }
      cout << endl;
    }
    }*/
  exit(0);

  //Perform cyclades
  Timer t;
  for (int i = 0; i < N_EPOCHS; i++) {
    vector<thread> threads;
    for (int j = 0; j < NTHREAD; j++) {
      threads.push_back(thread(do_cyclades_gradient_descent, ref(access_pattern[j]), ref(access_length[j]), ref(points), j));
    }
    for (int j = 0; j < NTHREAD; j++) {
      threads[j].join();
    }
    for (int j = 0; j < NTHREAD; j++) {
      thread_batch_on[j] = 0;
    }
    cout << compute_loss(points) << endl;;
  }
  cout << "ELAPSED TIME: " << t.elapsed() << endl;
  cout << "LOSS: " << compute_loss(points) << endl;;
}

int main(void) {
  for (int i = 0; i < RLENGTH; i++) {
    for (int j = 0; j < N_USERS; j++) {
      v_model[j][i] = 0;
    }
    for (int j = 0; j < N_MOVIES; j++) {
      u_model[i][j] = 0;
    }
  }
  cyclades_movielens_completion();
}
