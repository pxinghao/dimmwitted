#include <iostream>
#include "src/util.h"
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 
#include <math.h>
#include <time.h>
#include <map>
#include <unistd.h>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/adjacency_list.hpp>

#ifndef TIME_LIMIT
#define TIME_LIMIT 10000000
#endif

#ifndef CYC
#define CYC 0
#endif
#ifndef HOG
#define HOG 0
#endif

#define N_USERS 6041
#define N_MOVIES 3953

#define N_NUMA_NODES 2
#ifndef N_EPOCHS
#define N_EPOCHS 50
#endif
#define BATCH_SIZE 1000

#ifndef NTHREAD
#define NTHREAD 8
#endif

#define RLENGTH 30
#define GAMMA_REDUCTION_FACTOR .95
double GAMMA = 8e-2;
double C = 0;

using namespace std;

typedef tuple<int, int, double> DataPoint;

struct Comp
{
  bool operator()(const pair<int, int>& s1, const pair<int, int>& s2) {
    return s1.second > s2.second;
  }
};

int volatile thread_batch_on[NTHREAD];

//Initialize v and h matrix models
//double ** v_model, **rmodel;
double v_model[N_USERS][RLENGTH];
double u_model[N_MOVIES][RLENGTH];

double compute_loss(vector<DataPoint> &p) {
  double loss = 0;
  for (int i = 0; i < p.size(); i++) {
    int x = get<0>(p[i]),  y = get<1>(p[i]);
    double r = get<2>(p[i]), s = 0, dp = 0;
    for (int i = 0; i < RLENGTH; i++) {
      dp += v_model[x][i] * u_model[y][i];
    }
    double diff = dp - r - C;
    loss += diff * diff;
  }
  return sqrt(loss / (double)p.size());
}

void do_cyclades_gradient_descent(vector<int *> &access_pattern, vector<int> &access_length, vector<DataPoint> &points, int thread_id) {
  //Keep track of local model
  double local_v_model[N_USERS][RLENGTH];
  double local_u_model[N_MOVIES][RLENGTH];

  for (int batch = 0; batch < access_length.size(); batch++) {

    //Update local model
    /*for (int i = 0; i < RLENGTH; i++) {
      for (int j = 0; j < max(N_USERS, N_MOVIES); j++) {
	if (j < N_USERS) local_v_model[j][i] = v_model[j][i];
	if (j < N_MOVIES) local_u_model[j][i] = u_model[j][i];
      }
      }*/

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
	gradient += v_model[x][j] * u_model[y][j];
	//gradient += local_v_model[x][j] * local_u_model[y][j];
      }
      gradient -= r + C;

      //Perform updates
      for (int j = 0; j < RLENGTH; j++) {
	double new_v = v_model[x][j] - GAMMA * gradient * u_model[y][j];
	double new_u = u_model[y][j] - GAMMA * gradient * v_model[x][j];
	v_model[x][j] = new_v;
	u_model[y][j] = new_u;
	//double new_v = local_v_model[x][j] - GAMMA * gradient * local_u_model[y][j];
	//double new_u = local_u_model[y][j] - GAMMA * gradient * local_v_model[x][j];
	//local_v_model[x][j] = 24;
	//local_u_model[y][j] = 50;
      }
    }
  }
}

void do_cyclades_gradient_descent_no_sync(vector<int *> &access_pattern, vector<int> &access_length, vector<DataPoint> &points, int thread_id) {
  for (int batch = 0; batch < access_length.size(); batch++) {

    //For every data point in the connected component
    for (int i = 0; i < access_length[batch]; i++) {
      //Compute gradient
      DataPoint p = points[access_pattern[batch][i]];
      int x = get<0>(p), y = get<1>(p);
      double r = get<2>(p);
      double gradient = 0;
      for (int j = 0; j < RLENGTH; j++) {
	gradient += v_model[x][j] * u_model[y][j];
      }
      gradient -= r + C;

      //Perform updates
      for (int j = 0; j < RLENGTH; j++) {
	double new_v = v_model[x][j] - GAMMA * gradient * u_model[y][j];
	double new_u = u_model[y][j] - GAMMA * gradient * v_model[x][j];
	v_model[x][j] = new_v;
	u_model[y][j] = new_u;
      }
    }
  }
}

void distribute_ccs(map<int, vector<int> > &ccs, vector<vector<int *> > &access_pattern, vector<vector<int> > &access_length, int batchnum) {
  int chosen_threads[ccs.size()];
  int total_size_needed[NTHREAD];
  int count = 0;
  vector<pair<int, int> > balances;

  for (int i = 0; i < NTHREAD; i++) {
    total_size_needed[i] = 0;
    balances.push_back(pair<int, int>(i, 0));
  }

  //Count total size needed for each access pattern
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    pair<int, int> best_balance = balances.front();
    int chosen_thread = best_balance.first;
    chosen_threads[count] = chosen_thread;
    pop_heap(balances.begin(), balances.end(), Comp()); balances.pop_back();
    vector<int> cc = it->second;
    total_size_needed[chosen_thread] += cc.size();
    best_balance.second += cc.size();
    balances.push_back(best_balance); push_heap(balances.begin(), balances.end(), Comp());
  }

  //Allocate memory
  int index_count[NTHREAD];
  for (int i = 0; i < NTHREAD; i++) {
    int numa_node = i % N_NUMA_NODES;
    //numa_run_on_node(numa_node);
    //numa_set_localalloc();
    access_pattern[i][batchnum] = (int *)numa_alloc_onnode(total_size_needed[i] * sizeof(int), numa_node);
    //access_pattern[i][batchnum] = (int *)malloc(total_size_needed[i] * sizeof(int));
    access_length[i][batchnum] = total_size_needed[i];
    index_count[i] = 0;
  }

  //Copy memory over
  count = 0;  
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    vector<int> cc = it->second;
    int chosen_thread = chosen_threads[count];
    for (int i = 0; i < cc.size(); i++) {
      access_pattern[chosen_thread][batchnum][index_count[chosen_thread]+i] = cc[i];
    }
    index_count[chosen_thread] += cc.size();
  }
}

int union_find(int a, int *p) {
  int root = a;
  while ( p[a] != a) {
    a = p[a];
  }
  while ( root != a) {
    
    int root2 = p[root];
    p[root] = a;
    root = root2;
  }
  
  return a;
}

map<int, vector<int> > compute_CC(vector<DataPoint> &points, int start, int end) {
  //Numeric ordering:
  //Note: points.size() to mean end-start
  //data points - [0 ... points.size()]
  //user vars - [points.size() ... points.size() + N_USERS]
  //movie vars - [points.size() + N_USERS .... end]
  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> g (end-start + N_MOVIES + N_USERS);
  /*map<int, int> coordinate_id_map;
  int cur_coord_id = 0;

  for (int i = 0; i < end-start; i++) {
    boost::add_edge(i, end-start+get<0>(points[i+start]), g);
    boost::add_edge(i, end-start+N_USERS+get<1>(points[i+start]), g);
  }
  
  //CC
  vector<int> components(end-start + N_MOVIES + N_USERS);

  int num_total_components = boost::connected_components(g, &components[0]);*/
  int tree[end-start + N_MOVIES + N_USERS];
  for (int i = 0; i < end-start + N_MOVIES + N_USERS; i++) 
    tree[i] = i;
  for (int i = start; i < end; i++) {
    DataPoint p = points[i];
    int src = i-start;
    int e1 = get<0>(p) + end-start;
    int e2 = get<1>(p) + end-start+N_USERS;
    int c1 = union_find(src, tree);
    int c2 = union_find(e1, tree);
    int c3 = union_find(e2, tree);
    tree[c3] = c1;
    tree[c2] = c1;
  }
  
  map<int, vector<int> > CCs;

  //set<int> cids;
  for (int i = 0; i < end-start; i++) {
    CCs[union_find(i, tree)].push_back(i + start);
    //CCs[components[i]].push_back(i + start);
  }

  return CCs;
 }

void threaded_distribute_ccs(vector<DataPoint> &points, int start, int end, vector<vector<int *> > &access_pattern, vector<vector<int> > &access_length, int batchnum) {
  //Timer cc1;
  map<int, vector<int> > cc = compute_CC(points, start, end);
  //cout << cc1.elapsed() << endl;
  //cout << "COMPUTE CC " << cc1.elapsed() << endl;
  //Timer cc2;
  distribute_ccs(cc, access_pattern, access_length, batchnum);
  //cout << cc2.elapsed() << endl;
  //cout << " ALLOC " << cc2.elapsed() << endl;
}

vector<DataPoint> get_movielens_data() {
  vector<DataPoint> datapoints;

  ifstream in("ml-1m/ratings.dat");
  string s;
  string delimiter = "::";

  while (getline(in, s)) {
    int pos = 0;
    int count = 0;
    double features[4];
    while ((pos = s.find(delimiter)) != string::npos) {
      features[count++] = (double)stoi(s.substr(0, pos));
      s.erase(0, pos + delimiter.length());
    }
    C += features[2];
    datapoints.push_back(DataPoint(features[0], features[1], features[2]));
  }
  C /= (double)datapoints.size();
  return datapoints;
}

void cyclades_movielens_completion() {

  //Read data nonzero data points
  vector<DataPoint> points = get_movielens_data();
  random_shuffle(points.begin(), points.end());

  Timer overall;  

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
  Timer t2;
  vector<thread> ts;
  double cc_time = 0, alloc_time = 0;

  for (int i = 0; i < n_batches; i++) {
    
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());

    //ts.push_back(thread(threaded_distribute_ccs, ref(points), start, end, ref(access_pattern), ref(access_length), i));
    //threaded_distribute_ccs(points, start, end, access_pattern, access_length, i);
    ///Compute connected components of data points
    //Timer ttt;
    map<int, vector<int> > cc = compute_CC(points, start, end);
    //cc_time += ttt.elapsed();

    //Distribute connected components across threads
    //Timer ttt2;
    distribute_ccs(cc, access_pattern, access_length, i);
    //alloc_time += ttt2.elapsed();
  }
  //cout << "CYCLADES CC ALLOC TIME: " << t2.elapsed() << endl;
  for (int i = 0; i < ts.size(); i++) ts[i].join();
  //cout << cc_time << " " << alloc_time << endl;

  /*int num_work = 0;
  //for (int j = 0; j < n_batches; j++) {
  for (int j = 0; j < 1; j++) {
      cout << "BATCH " << j << endl;
    for (int i = 0; i < NTHREAD; i++) {
      cout << access_length[i][j] << " : ";
      cout << "THREAD " << i << endl;
      num_work += access_length[i][j];
      for (int k = 0; k < min(10, access_length[i][j]); k++) {
	cout << access_pattern[i][j][k] << " ";
      }
      cout << endl;
    }
  }
  cout << "TOT WORK " << num_work << endl;
  exit(0);*/

  //Perform cyclades
  Timer gradient_time;
  for (int i = 0; i < N_EPOCHS; i++) {
    vector<thread> threads;
    for (int j = 0; j < NTHREAD; j++) {
      numa_run_on_node(j % N_NUMA_NODES);
      threads.push_back(thread(do_cyclades_gradient_descent, ref(access_pattern[j]), ref(access_length[j]), ref(points), j));
    }
    for (int j = 0; j < threads.size(); j++) {
      threads[j].join();
    }
    for (int j = 0; j < NTHREAD; j++) {
      thread_batch_on[j] = 0;
    }
    GAMMA *= GAMMA_REDUCTION_FACTOR;
    //cout << i << " " << compute_loss(points) << " " << overall.elapsed() << endl;
    //cout << i << " " << compute_loss(points) << " " << gradient_time.elapsed() << endl;
  }
  cout << "CYCLADES OVERALL TIME: " << overall.elapsed() << endl;
  cout << "CYCLADES GRADIENT TIME: " << gradient_time.elapsed() << endl;
  cout << "LOSS: " << compute_loss(points) << endl;;
  //cout << overall.elapsed() << endl;
  //cout << gradient_time.elapsed() << endl;
  //cout << compute_loss(points) << endl;
}

void hogwild_completion() {
  //Get data
  vector<DataPoint> points = get_movielens_data();
  random_shuffle(points.begin(), points.end());

  Timer overall;

  //Hogwild access pattern construction
  vector<vector<int *> > access_pattern(NTHREAD);
  vector<vector<int > > access_length(NTHREAD);

  //Timer t2;
  int n_points_per_thread = points.size() / NTHREAD;
  for (int i = 0; i < NTHREAD; i++) {
    //No batches in hogwild, so treat it as all 1 batch
    access_pattern[i].resize(1);
    access_length[i].resize(1);

    int start = i * n_points_per_thread;
    int end = min(i * n_points_per_thread + n_points_per_thread, (int)points.size());

    access_pattern[i][0] = (int *)malloc(sizeof(int) * n_points_per_thread);
    for (int j = start; j < end; j++) {
      access_pattern[i][0][j-start] = j;
    }
    access_length[i][0] = n_points_per_thread;
  }
  //cout << "ALLOCATION TIME " << t2.elapsed() << endl;

  /*int num_work = 0;
  //for (int j = 0; j < n_batches; j++) {
  for (int j = 0; j < 1; j++) {
      cout << "BATCH " << j << endl;
    for (int i = 0; i < NTHREAD; i++) {
      cout << access_length[i][j] << " : ";
      cout << "THREAD " << i << endl;
      num_work += access_length[i][j];
      for (int k = 0; k < min(10, access_length[i][j]); k++) {
	cout << access_pattern[i][j][k] << " ";
      }
      cout << endl;
    }
  }
  cout << "TOT WORK " << num_work << endl;
  exit(0);*/

  //Divide to threads
  Timer gradient_time;
  for (int i = 0; i < N_EPOCHS; i++) {
    vector<thread> threads;
    for (int j = 0; j < NTHREAD; j++) {
      threads.push_back(thread(do_cyclades_gradient_descent_no_sync, ref(access_pattern[j]), ref(access_length[j]), ref(points), j));
    }
    for (int j = 0; j < threads.size(); j++) {
      threads[j].join();
    }
    GAMMA *= GAMMA_REDUCTION_FACTOR;
    //cout << i << " " << compute_loss(points) << " " << overall.elapsed() << endl;
    //cout << i << " " << compute_loss(points) << " " << gradient_time.elapsed() << endl;
  }
  //cout << overall.elapsed() << endl;
  cout << "HOGWILD OVERALL TIME: " << overall.elapsed() << endl;
  cout << "HOGWILD GRADIENT TIME: " << gradient_time.elapsed() << endl;
  cout << "LOSS: " << compute_loss(points) << endl;
  //cout << gradient_time.elapsed() << endl;
  //cout << compute_loss(points) << endl;
}

int main(void) {
  srand(0);
  for (int i = 0; i < RLENGTH; i++) {
    for (int j = 0; j < N_USERS; j++) {
      v_model[j][i] = ((double)rand()/(double)RAND_MAX);
    }
    for (int j = 0; j < N_MOVIES; j++) {
      u_model[j][i] = ((double)rand()/(double)RAND_MAX);
    }
  }
  if (CYC)
    cyclades_movielens_completion();
  if (HOG)
    hogwild_completion();
}
