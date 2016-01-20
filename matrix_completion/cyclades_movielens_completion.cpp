#include <iostream>
#include <stack>
#include "../src/util.h"
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 
#include <math.h>
#include <time.h>
#include <map>
#include <unistd.h>
#include <set>
#include <numa.h>
#include <sched.h>

#ifndef TIME_LIMIT
#define TIME_LIMIT 10000000
#endif

#ifndef CYC
#define CYC 0
#endif
#ifndef HOG
#define HOG 0
#endif

//#define N_USERS 6041 //1M dataset
//#define N_MOVIES 3091 //1M dataset
#define N_USERS 71568
#define N_MOVIES 10682

#define N_NUMA_NODES 2
#ifndef N_EPOCHS
#define N_EPOCHS 80
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 5000
#endif

#ifndef NTHREAD
#define NTHREAD 1
#endif

#ifndef RLENGTH
#define RLENGTH 1
#endif

#if HOG == 1
#undef SHOULD_SYNC
#define SHOULD_SYNC 0
#endif
#ifndef SHOULD_SYNC
#define SHOULD_SYNC 1
#endif

#ifndef SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH
#define SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH 0
#endif

#ifndef MOD_REP_CYC
#define MOD_REP_CYC 0
#endif

#ifndef REGULARIZE
#define REGULARIZE 1
#endif

#ifndef CRIMP
#define CRIMP 0
#endif

#define GAMMA_REDUCTION_FACTOR 1

double GAMMA = 5e-5;
double ALPHA = 1 / (double)(RLENGTH * (N_MOVIES + N_USERS));
double C = 0;

using namespace std;

typedef tuple<int, int, double> DataPoint;

struct Comp
{
  bool operator()(const pair<int, int>& s1, const pair<int, int>& s2) {
    return s1.second > s2.second;
  }
};

//int regularization_bookkeeping[
int volatile thread_batch_on[NTHREAD];
int volatile thread_batch_on2[NTHREAD];

//Bookkeeping
int bookkeeping_v[N_USERS], bookkeeping_u[N_MOVIES];

//Initialize v and h matrix models
//double ** v_model __attribute__((aligned(64))), **u_model __attribute__((aligned(64)));

double ** v_model_records[N_EPOCHS];
double ** u_model_records[N_EPOCHS];
double gradient_times[N_EPOCHS], overall_times[N_EPOCHS];

#define num_doubles_in_cacheline  (64 / sizeof(double))
#define cache_align_users  ((N_USERS / num_doubles_in_cacheline+1) * num_doubles_in_cacheline)
#define cache_align_rlength  ((RLENGTH / num_doubles_in_cacheline+1) * num_doubles_in_cacheline)
#define cache_align_n_movies  ((N_MOVIES / num_doubles_in_cacheline+1) * num_doubles_in_cacheline)
#define cache_length_rlength ((RLENGTH / 8 + 1) * 8)

double v_model[N_USERS][cache_length_rlength] __attribute__((aligned(64)));
double u_model[N_MOVIES][cache_length_rlength] __attribute__((aligned(64)));

//double local_vv_model[NTHREAD][N_USERS][RLENGTH];
//double local_uu_model[NTHREAD][N_MOVIES][RLENGTH];
//double local_vv_model[NTHREAD][cache_align_users][cache_align_rlength] __attribute__((aligned(64)));
//double local_uu_model[NTHREAD][cache_align_users][cache_align_rlength] __attribute__((aligned(64)));;
double **local_vv_model[NTHREAD]  __attribute__((aligned(64)));
double **local_uu_model[NTHREAD]  __attribute__((aligned(64)));

double thread_load_balance[NTHREAD];
double load_balance_avg_max = 0;

size_t cur_bytes_allocated[NTHREAD];
int cur_datapoints_used[NTHREAD];
int core_to_node[NTHREAD];

int iterate = 0;

int volatile finished_computation = 0;

void pin_to_core(size_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void clear_bookkeeping() {
  for (int i = 0; i < N_USERS; i++) bookkeeping_v[i] = 0;
  for (int i = 0; i < N_MOVIES; i++) bookkeeping_u[i] = 0;
}

double compute_loss(vector<DataPoint> &p) {
  double loss = 0;
  for (int i = 0; i < p.size(); i++) {
    int x = get<0>(p[i]),  y = get<1>(p[i]);
    double r = get<2>(p[i]), s = 0, dp = 0;
    for (int j = 0; j < RLENGTH; j++) {
      dp += v_model[x][j] * u_model[y][j];
    }
    double diff = dp - r - C;
    loss += diff * diff;
  }
  return sqrt(loss / (double)p.size());
}

double compute_loss_regularize(vector<DataPoint> &p) {
  double loss = 0;
  for (int i = 0; i < p.size(); i++) {
    int x = get<0>(p[i]),  y = get<1>(p[i]);
    double r = get<2>(p[i]), s = 0, dp = 0;
    for (int j = 0; j < RLENGTH; j++) {
      dp += v_model[x][j] * u_model[y][j];
    }
    double diff = dp - r - C;
    loss += diff * diff;
  }

  double loss2 = 0, loss3 = 0;
  for (int i = 0; i < N_USERS; i++) {
    for (int k = 0; k < RLENGTH; k++) {
      loss2 += v_model[i][k] * v_model[i][k];
    }
  }
  for (int i = 0; i < N_MOVIES; i++) {
    for (int k = 0; k < RLENGTH; k++) {
      loss2 += u_model[i][k] * u_model[i][k];
    }
  }

  return sqrt(loss/(double)p.size()) + ALPHA * (loss2 + loss3);
}

double compute_loss_for_record_epoch(vector<DataPoint> &p, int epoch) {
  double loss = 0;
  for (int i = 0; i < p.size(); i++) {
    int x = get<0>(p[i]),  y = get<1>(p[i]);
    double r = get<2>(p[i]), s = 0, dp = 0;
    for (int j = 0; j < RLENGTH; j++) {
      dp += v_model_records[epoch][x][j] * u_model_records[epoch][y][j];
    }
    double diff = dp - r - C;
    loss += diff * diff;
  }
  return sqrt(loss / (double)p.size());
}

double compute_loss_regularize_for_record_epoch(vector<DataPoint> &p, int epoch) {
  double loss = 0;
  for (int i = 0; i < p.size(); i++) {
    int x = get<0>(p[i]),  y = get<1>(p[i]);
    double r = get<2>(p[i]), s = 0, dp = 0;
    for (int j = 0; j < RLENGTH; j++) {
      dp += v_model_records[epoch][x][j] * u_model_records[epoch][y][j];
    }
    double diff = dp - r - C;
    loss += diff * diff;
  }

  double loss2 = 0, loss3 = 0;
  for (int i = 0; i < N_USERS; i++) {
    for (int k = 0; k < RLENGTH; k++) {
      loss2 += v_model_records[epoch][i][k] * v_model_records[epoch][i][k];
    }
  }
  for (int i = 0; i < N_MOVIES; i++) {
    for (int k = 0; k < RLENGTH; k++) {
      loss2 += u_model_records[epoch][i][k] * u_model_records[epoch][i][k];
    }
  }

  return sqrt(loss / (double)p.size()) + ALPHA * (loss2 + loss3);
}

double print_loss_for_records(vector<DataPoint> &p) {
  for (int i = 0; i < N_EPOCHS; i++) {
    double loss;
    if (REGULARIZE)
      loss = compute_loss_regularize_for_record_epoch(p, i);
    else
      loss = compute_loss_for_record_epoch(p, i);
    double overall_time = overall_times[i];
    double gradient_time = gradient_times[i];
    cout << loss << " " << overall_time << " " << gradient_time << endl;
  }
}

double copy_model_to_records(int epoch, double overall_time, double gradient_time) {
  gradient_times[epoch] = gradient_time;
  overall_times[epoch] = overall_time;
  for (int i = 0; i < N_USERS; i++) {
    for (int j = 0; j < RLENGTH; j++) {
      v_model_records[epoch][i][j] = v_model[i][j];
    }
  }
  for (int i = 0; i < N_MOVIES; i++) {
    for (int j = 0; j < RLENGTH; j++) {
      u_model_records[epoch][i][j] = u_model[i][j];
    }
  }
}

void do_cyclades_gradient_descent_with_points(DataPoint * access_pattern, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, int thread_id) {
  pin_to_core(thread_id);

  for (int batch = 0; batch < access_length.size(); batch++) {
    
    if (SHOULD_SYNC) {
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
    }
    
    //For every data point in the connected component
    for (int i = 0; i < access_length[batch]; i++) {
      //Compute gradient
      DataPoint p = access_pattern[batch_index_start[batch]+i];
      //DataPoint p = access_pattern[i];
      int x = get<0>(p), y = get<1>(p);
      double r = get<2>(p);            
      double gradient = 0;
      int update_order = order[batch_index_start[batch]+i];
      
      for (int j = 0; j < RLENGTH; j++) {
	gradient += v_model[x][j] * u_model[y][j];
      }
      gradient -= r + C;

      double v_reg_param = 1, u_reg_param = 1;
      if (REGULARIZE) {
	double v_diff = update_order - bookkeeping_v[x] - 1;
	double u_diff = update_order - bookkeeping_u[y] - 1;
	v_reg_param = pow((1-2*ALPHA*GAMMA), v_diff);
	u_reg_param = pow((1-2*ALPHA*GAMMA), u_diff);
      }

      //Perform updates
      for (int j = 0; j < RLENGTH; j++) {
	v_model[x][j] = v_reg_param * v_model[x][j];
	u_model[y][j] = u_reg_param * u_model[y][j];
	double new_v = v_model[x][j] - GAMMA * gradient * u_model[y][j];
	double new_u = u_model[y][j] - GAMMA * gradient * v_model[x][j];
	v_model[x][j] = new_v;
	u_model[y][j] = new_u;
      }

      //Update bookkeeping
      if (REGULARIZE) {
	bookkeeping_v[x] = update_order;
	bookkeeping_u[y] = update_order;
      }
    }
  }
}

void do_cyclades_gradient_descent_with_points_mod_rep(DataPoint * access_pattern, vector<int> &access_length, vector<int> &batch_index_start, int thread_id) {
  pin_to_core(thread_id);

  //double local_v_model[N_USERS][RLENGTH];
  //double local_u_model[N_MOVIES][RLENGTH];

  for (int batch = 0; batch < access_length.size(); batch++) {

    
    if (SHOULD_SYNC) {
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
    }
    
    //For every data point in the connected component
    for (int i = 0; i < access_length[batch]; i++) {
      //Compute gradient
      DataPoint p = access_pattern[batch_index_start[batch]+i];
      //DataPoint p = access_pattern[i];
      int x = get<0>(p), y = get<1>(p);
      double r = get<2>(p);            
      double gradient = 0;
      
      for (int j = 0; j < RLENGTH; j++) {
	//gradient += local_v_model[x][j] * local_u_model[y][j];
	gradient += local_vv_model[thread_id][x][j] * local_uu_model[thread_id][y][j];
	//gradient += v_model[x][j] * u_model[y][j];
      }
      gradient -= r + C;

      //Perform updates
      for (int j = 0; j < RLENGTH; j++) {
	//double new_v = local_v_model[x][j] - GAMMA * gradient * local_u_model[y][j];
	//double new_u = local_u_model[y][j] - GAMMA * gradient * local_v_model[x][j];
	//local_v_model[x][j] = new_v;
	//local_u_model[y][j] = new_u;
	
	double new_v = local_vv_model[thread_id][x][j] - GAMMA * gradient * local_uu_model[thread_id][y][j];
	double new_u = local_uu_model[thread_id][y][j] - GAMMA * gradient * local_vv_model[thread_id][x][j];
	local_vv_model[thread_id][x][j] = new_v;
	local_uu_model[thread_id][y][j] = new_u;
	//v_model[x][j] = new_v;
	//u_model[y][j] = new_u;
      }
    }

    /*for (int i = 0; i < access_length[batch]; i++) {
      DataPoint p = access_pattern[batch_index_start[batch]+i];
      //DataPoint p = access_pattern[i];
      int x = get<0>(p), y = get<1>(p);
      for (int j = 0; j < RLENGTH; j++){
	v_model[x][j] = local_vv_model[thread_id][x][j];
	u_model[y][j] = local_uu_model[thread_id][y][j];
      }
      }*/
  }
}

void distribute_ccs(map<int, vector<int> > &ccs, vector<DataPoint *> &access_pattern, vector<vector<int> > &access_length, vector<vector<int> > &batch_index_start, int batchnum, vector<DataPoint> &points, vector<vector<int> > &order) {
  
  int * chosen_threads = (int *)malloc(sizeof(int) * ccs.size());
  int total_size_needed[NTHREAD];
  int count = 0;
  vector<pair<int, int> > balances;

  for (int i = 0; i < NTHREAD; i++) {
    total_size_needed[i] = 0;
    balances.push_back(pair<int, int>(i, 0));
  }

  //Count total size needed for each access pattern
  double max_cc_size = 0, avg_cc_size = 0;
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    pair<int, int> best_balance = balances.front();
    int chosen_thread = best_balance.first;
    chosen_threads[count] = chosen_thread;
    pop_heap(balances.begin(), balances.end(), Comp()); balances.pop_back();
    vector<int> cc = it->second;
    total_size_needed[chosen_thread] += cc.size();
    best_balance.second += cc.size();
    balances.push_back(best_balance); push_heap(balances.begin(), balances.end(), Comp());
    max_cc_size = max(max_cc_size, (double) cc.size());
    avg_cc_size += cc.size();
  }
  avg_cc_size /= (double)ccs.size();

  //Allocate memory
  int index_count[NTHREAD];
  int max_load = 0;
  for (int i = 0; i < NTHREAD; i++) {
    //int numa_node = i % 2;
    int numa_node = core_to_node[i];

    batch_index_start[i][batchnum] = cur_datapoints_used[i];
    if (cur_bytes_allocated[i] == 0) {
      size_t new_size = (size_t)total_size_needed[i] * sizeof(DataPoint);
      new_size = ((new_size / numa_pagesize()) + 1) * numa_pagesize();
      access_pattern[i] = (DataPoint *)numa_alloc_onnode(new_size, numa_node);
      cur_bytes_allocated[i] = new_size;
    }
    else {
      if ((cur_datapoints_used[i] + total_size_needed[i])*sizeof(DataPoint) >=
	  cur_bytes_allocated[i]) {
	size_t old_size = (size_t)(cur_bytes_allocated[i]);
	size_t new_size = (size_t)((cur_datapoints_used[i] + total_size_needed[i]) * sizeof(DataPoint));
	//Round new size to next page length 
	new_size = ((new_size / numa_pagesize()) + 1) * numa_pagesize();
	
	//DataPoint *new_mem = (DataPoint *)numa_alloc_onnode(new_size, numa_node);
	//memcpy(new_mem, access_pattern[i], old_size);
	//numa_free(access_pattern[i], old_size);
	//access_pattern[i] = new_mem;
	//access_pattern[i] = (DataPoint *)numa_alloc_onnode(new_size, numa_node);
	access_pattern[i] = (DataPoint *)numa_realloc(access_pattern[i], old_size, new_size);
	//access_pattern[i] = (DataPoint *)realloc(access_pattern[i], new_size);

	cur_bytes_allocated[i] = new_size;
      }
    }
    cur_datapoints_used[i] += total_size_needed[i];
    order[i].resize(cur_datapoints_used[i]);
    if (access_pattern[i] == NULL) {
      cout << "OOM" << endl;
      exit(0);
    }
      
    access_length[i][batchnum] = total_size_needed[i];
    index_count[i] = 0;
    thread_load_balance[i] += total_size_needed[i];
  }

  //Copy memory over
  count = 0;  
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    vector<int> cc = it->second;
    int chosen_thread = chosen_threads[count];
    for (int i = 0; i < cc.size(); i++) {
      access_pattern[chosen_thread][index_count[chosen_thread]+batch_index_start[chosen_thread][batchnum] + i] = points[cc[i]];
      order[chosen_thread][index_count[chosen_thread]+batch_index_start[chosen_thread][batchnum] + i] = cc[i]+1;
    }
    index_count[chosen_thread] += cc.size();
  }
  free(chosen_threads);
}

int union_find(int a, int *p) {
  int root = a;
  while (p[a] != a) {
    a = p[a];
  }
  while (root != a) {    
    int root2 = p[root];
    p[root] = a;
    root = root2;
  }
  
  return a;
}

map<int, vector<int> > compute_CC(vector<DataPoint> &points, int start, int end) {
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
  for (int i = 0; i < end-start; i++) {
    int group = union_find(i, tree);
    CCs[group].push_back(i+start);    
  }
  return CCs;
 }

vector<DataPoint> get_movielens_data() {
  vector<DataPoint> datapoints;

  ifstream in("ml-10M100K/ratings.dat");
  //ifstream in("ml-1m/ratings.dat");
  string s;
  string delimiter = "::";
  //int m1 = 0, m2 = 0;
  set<int> valid_remapped_ids;
  for (int i = 0; i < N_MOVIES; i++) {
    valid_remapped_ids.insert(i);
  }
  while (getline(in, s)) {
    int pos = 0;
    int count = 0;
    double features[4];
    while ((pos = s.find(delimiter)) != string::npos) {
      features[count++] = (double)stoi(s.substr(0, pos));
      s.erase(0, pos + delimiter.length());
    }
    C += features[2];
    valid_remapped_ids.erase((int)features[1]);
    //m1=max(m1,(int)features[0]);
    //m2=max(m2,(int)features[1]);
    datapoints.push_back(DataPoint(features[0], features[1], features[2]));
  }

  //Renormalize movie data (param 1)
  //Renormalize movie data (param 1)
  set<int>::iterator cur_id = valid_remapped_ids.begin();
  map<int, int> mappings;
  for (int i = 0; i < datapoints.size(); i++) {
    if (get<1>(datapoints[i]) >= N_MOVIES) {
      if (mappings.find(get<1>(datapoints[i])) != mappings.end()) {
	get<1>(datapoints[i]) = mappings[get<1>(datapoints[i])];
      }
      else {
	int prev = get<1>(datapoints[i]);
	get<1>(datapoints[i]) = *cur_id;
	mappings[prev] = get<1>(datapoints[i]);
	cur_id++;
      }
    }
  }

  //cout << m1 << " " << m2 << endl;
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
  vector<DataPoint *> access_pattern(NTHREAD);
  //Access length (number of elements in corresponding cc)
  vector<vector<int > > access_length(NTHREAD);
  vector<vector<int > > batch_index_start(NTHREAD);
  vector<vector<int> > order(NTHREAD);

  for (int i = 0; i < NTHREAD; i++) {
    //access_pattern[i].resize(n_batches);
    access_length[i].resize(n_batches);
    batch_index_start[i].resize(n_batches);
    order[i].resize(n_batches);
  }

  //CC Generation
  Timer t2;
  vector<thread> ts;
  double cc_time = 0, alloc_time = 0; 

  for (int i = 0; i < n_batches; i++) {
    
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());

    ///Compute connected components of data points
    Timer ttt;
    map<int, vector<int> > cc = compute_CC(points, start, end);
    cc_time += ttt.elapsed();
    //Distribute connected components across threads
    Timer ttt2;
    distribute_ccs(cc, access_pattern, access_length, batch_index_start, i, points, order);
    alloc_time += ttt2.elapsed();
  }

  //cout << "CYCLADES CC ALLOC TIME: " << t2.elapsed() << endl;
  for (int i = 0; i < ts.size(); i++) ts[i].join();

  //Perform cyclades
  float copy_model_time = 0;
  Timer gradient_time;

  for (int i = 0; i < N_EPOCHS; i++) {
    //cout << compute_loss_regularize(points) << endl;
    if (REGULARIZE) {
      clear_bookkeeping();
    }
    vector<thread> threads;
    if (NTHREAD == 1) {
      do_cyclades_gradient_descent_with_points(access_pattern[0], access_length[0], batch_index_start[0], order[0], 0);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, ref(access_pattern[j]), ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j));      
      }
      for (int j = 0; j < threads.size(); j++) {
	threads[j].join();
      }
      for (int j = 0; j < NTHREAD; j++) {
	thread_batch_on[j] = 0;
      }
    }
    GAMMA *= GAMMA_REDUCTION_FACTOR;
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
      Timer copy_model_timer;
      copy_model_to_records(i, overall.elapsed()-copy_model_time, gradient_time.elapsed()-copy_model_time);
      copy_model_time += copy_model_timer.elapsed();
    }
  }

  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    if (REGULARIZE)
      cout << compute_loss_regularize(points) << endl;
    else
      cout << compute_loss(points) << endl;
  }
  else {
    print_loss_for_records(points);
  }
}

void cyclades_movielens_completion_mod_rep() {

  //Read data nonzero data points
  vector<DataPoint> points = get_movielens_data();
  random_shuffle(points.begin(), points.end());
  Timer overall;  

  //Access pattern generation
  int n_batches = (int)ceil((points.size() / (double)BATCH_SIZE));

  //Access pattern of form [thread][batch][point_ids]
  vector<DataPoint *> access_pattern(NTHREAD);
  //Access length (number of elements in corresponding cc)
  vector<vector<int > > access_length(NTHREAD);
  vector<vector<int > > batch_index_start(NTHREAD);
  vector<vector<int> > order(NTHREAD);

  for (int i = 0; i < NTHREAD; i++) {
    //access_pattern[i].resize(n_batches);
    access_length[i].resize(n_batches);
    batch_index_start[i].resize(n_batches);
    order[i].resize(n_batches);
  }

  //CC Generation
  Timer t2;
  vector<thread> ts;
  double cc_time = 0, alloc_time = 0; 

  for (int i = 0; i < n_batches; i++) {
    
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());

    ///Compute connected components of data points
    Timer ttt;
    map<int, vector<int> > cc = compute_CC(points, start, end);
    cc_time += ttt.elapsed();
    //Distribute connected components across threads
    Timer ttt2;
    distribute_ccs(cc, access_pattern, access_length, batch_index_start, i, points, order);
    alloc_time += ttt2.elapsed();
  }

  //cout << "CYCLADES CC ALLOC TIME: " << t2.elapsed() << endl;
  for (int i = 0; i < ts.size(); i++) ts[i].join();

  //Perform cyclades
  Timer gradient_time;

  for (int i = 0; i < N_EPOCHS; i++) {
    if (REGULARIZE) {
      clear_bookkeeping();
    }
    vector<thread> threads;
    if (NTHREAD == 1) {
      do_cyclades_gradient_descent_with_points(access_pattern[0], access_length[0], batch_index_start[0], order[0], 0);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points_mod_rep, ref(access_pattern[j]), ref(access_length[j]), ref(batch_index_start[j]), j));
      }
      for (int j = 0; j < threads.size(); j++) {
	threads[j].join();
      }
      for (int j = 0; j < NTHREAD; j++) {
	thread_batch_on[j] = 0;
      }
    }
    GAMMA *= GAMMA_REDUCTION_FACTOR;
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) copy_model_to_records(i, overall.elapsed(), gradient_time.elapsed());
  }
  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    if (REGULARIZE)
      cout << compute_loss_regularize(points) << endl;
    else
      cout << compute_loss(points) << endl;
  }
  else {
    print_loss_for_records(points);
  }
}

void hogwild_completion() {
  //Get data
  vector<DataPoint> points = get_movielens_data();
  random_shuffle(points.begin(), points.end());

  Timer overall;

  //Hogwild access pattern construction
  vector<DataPoint *> access_pattern(NTHREAD);
  vector<vector<int > > access_length(NTHREAD);
  vector<vector<int> > batch_index_start(NTHREAD);
  vector<vector<int> > order(NTHREAD);

  //Timer t2;
  int n_points_per_thread = points.size() / NTHREAD;
  for (int i = 0; i < NTHREAD; i++) {
    //No batches in hogwild, so treat it as all 1 batch
    access_length[i].resize(1);
    batch_index_start[i].resize(1);

    int start = i * n_points_per_thread;
    int end = min(i * n_points_per_thread + n_points_per_thread, (int)points.size());

    batch_index_start[i][0] = 0;
    access_pattern[i] = (DataPoint *)malloc(sizeof(DataPoint) * n_points_per_thread);
    order[i].resize(n_points_per_thread);
    for (int j = start; j < end; j++) {
      access_pattern[i][j-start] = points[j];
      order[i][j-start] = j+1;
    }
    access_length[i][0] = n_points_per_thread;
  }

  //Divide to threads
  float copy_model_time = 0;
  Timer gradient_time;

  for (int i = 0; i < N_EPOCHS; i++) {
    //cout << compute_loss_regularize(points) << endl;
    if (REGULARIZE) {
      clear_bookkeeping();
    }
    vector<thread> threads;
    if (NTHREAD == 1) {
      do_cyclades_gradient_descent_with_points(access_pattern[0], access_length[0], batch_index_start[0], order[0], 0);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, ref(access_pattern[j]), ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j));
      }
      for (int j = 0; j < threads.size(); j++) {
	threads[j].join();
      }
    }
    GAMMA *= GAMMA_REDUCTION_FACTOR;
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
      Timer copy_model_timer;
      copy_model_to_records(i, overall.elapsed()-copy_model_time, gradient_time.elapsed()-copy_model_time);
      copy_model_time += copy_model_timer.elapsed();
    }
  }

  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    if (REGULARIZE)
      cout << compute_loss_regularize(points) << endl;
    else
      cout << compute_loss(points) << endl;
  }
  else {
    print_loss_for_records(points);
  }
}

int main(void) {
  setprecision(15);
  pin_to_core(0);
  //Create a map from core/thread -> node
  for (int i = 0; i < NTHREAD; i++) core_to_node[i] = -1;
  int num_cpus = numa_num_task_cpus();
  struct bitmask *bm = numa_bitmask_alloc(num_cpus);
  for (int i=0; i<=numa_max_node(); ++i) {
      numa_node_to_cpus(i, bm);
      for (int j = 0; j < min((int)bm->size, NTHREAD); j++) {
	if (numa_bitmask_isbitset(bm, j)) {
	  core_to_node[j] = i;
	}
      }
    }

  for (int i = 0; i < NTHREAD; i++) {
    cur_bytes_allocated[i] = 0;
    cur_datapoints_used[i] = 0;
  }

  if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    for (int i = 0; i < N_EPOCHS; i++) {
      v_model_records[i] = (double **)malloc(sizeof(double *) * N_USERS);
      u_model_records[i] = (double **)malloc(sizeof(double *) * N_MOVIES);
      for (int j = 0; j < N_USERS; j++) 
	v_model_records[i][j] = (double *)malloc(sizeof(double) * RLENGTH);
      for (int j = 0; j < N_MOVIES; j++) 
	u_model_records[i][j] = (double *)malloc(sizeof(double) * RLENGTH);
    }
  }

  /*for (int i = 0; i < NTHREAD; i++) {
    local_vv_model[i] = (double **)numa_alloc_onnode(sizeof(double *)*cache_align_users, core_to_node[i]);
    local_uu_model[i] = (double **)numa_alloc_onnode(sizeof(double *)*cache_align_n_movies, core_to_node[i]);
    for (int j = 0; j < cache_align_users; j++) 
      local_vv_model[i][j] = (double *)numa_alloc_onnode(sizeof(double)*cache_align_rlength, core_to_node[i]);
    for (int j = 0; j < cache_align_n_movies; j++)
      local_uu_model[i][j ] = (double *)numa_alloc_onnode(sizeof(double)*cache_align_rlength, core_to_node[i]);
  }

  /*v_model = (double **)numa_alloc_onnode(sizeof(double *)*cache_align_users, 0);
  u_model = (double **)numa_alloc_onnode(sizeof(double *)*cache_align_n_movies, 0);
  for (int j = 0; j < cache_align_users; j++) 
    v_model[j] = (double *)numa_alloc_onnode(sizeof(double)*cache_align_rlength, 0);
  for (int j = 0; j < cache_align_n_movies; j++)
  u_model[j] = (double *)numa_alloc_onnode(sizeof(double)*cache_align_rlength, 0);*/
  
  srand(0);
  for (int i = 0; i < RLENGTH; i++) {
    for (int j = 0; j < N_USERS; j++) {
      v_model[j][i] = ((double)rand()/(double)RAND_MAX);
    }
    for (int j = 0; j < N_MOVIES; j++) {
      u_model[j][i] = ((double)rand()/(double)RAND_MAX);
    }
  }

  if (MOD_REP_CYC)
    cyclades_movielens_completion_mod_rep();
  if (CYC)
    cyclades_movielens_completion();
  if (HOG)
    hogwild_completion();
}
