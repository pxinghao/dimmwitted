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
#include <iomanip> 
#include <mutex> 
#include <omp.h>

#define WORD_EMBEDDINGS_FILE "/run/sparse_graph"
//#define N_NODES 628
//#define N_DATAPOINTS 5607
//#define N_NODES 3822
//#define N_DATAPOINTS 80821
//#define N_NODES 213271
//#define N_DATAPOINTS 20207156
//#define N_NODES 16774
//#define N_DATAPOINTS 622941
//#define N_NODES 105
//#define N_DATAPOINTS 1096
//HUGE GRAPH
#define N_NODES 1468307
#define N_DATAPOINTS 140196115

#ifndef NTHREAD
#define NTHREAD 8
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 200
#endif 

#ifndef BATCH_SIZE
//#define BATCH_SIZE 4250 //full 80 mb
#define BATCH_SIZE 6000 //full 1 GB
//#define BATCH_SIZE 520
//#define BATCH_SIZE 2000 //1/10 of 80 mb SGD
//#define BATCH_SIZE 490
#endif

#ifndef HOG
#define HOG 0
#endif

#ifndef CYC
#define CYC 0
#endif

#ifndef SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH
#define SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH 1
#endif

#if HOG == 1
#undef SHOULD_SYNC
#define SHOULD_SYNC 0
#endif
#ifndef SHOULD_SYNC
#define SHOULD_SYNC 1
#endif

#ifndef K
#define K 100
#endif
#define K_TO_CACHELINE ((K / 8 + 1) * 8)

#ifndef SAGA
#define SAGA 0
#endif

#ifndef START_GAMMA
//#define START_GAMMA 2.3e-4//3.42e-5
#define START_GAMMA 1e-10 // SGD CYC/HOG
//#define START_GAMMA 1e-7 // SAGA CYC
//#define START_GAMMA 3e-11 // SAGA HOG
//#define START_GAMMA 1e-10 //SAGA HOG DIVERGE
//#define START_GAMMA 1e-10 // SG

//#define START_GAMMA 1e-4 //Test
#endif

double volatile C = 0;
double GAMMA = START_GAMMA;
//double GAMMA = 3e-4;
//double GAMMA = 8e-4;
double GAMMA_REDUCTION = 1;

int volatile thread_batch_on[NTHREAD];

//double sum_gradients[N_NODES][K_TO_CACHELINE] __attribute__((aligned(64)));
//double prev_gradients[N_DATAPOINTS][2][K_TO_CACHELINE] __attribute__((aligned(64)));
//double model[N_NODES][K_TO_CACHELINE] __attribute__((aligned(64)));
//double prevv_gradients[N_DATAPOINTS][2][K_TO_CACHELINE];
double *C_sum_mult[NTHREAD];
double *C_sum_mult2[NTHREAD];
double ** prev_gradients[NTHREAD];
double ** model, **sum_gradients;
double **model_records[N_EPOCHS];
int *tree_thread[NTHREAD];
int bookkeeping[N_NODES];

double gradient_times[N_EPOCHS], overall_times[N_EPOCHS];
double thread_load_balance[NTHREAD];
size_t cur_bytes_allocated[NTHREAD];
int cur_datapoints_used[NTHREAD];
int core_to_node[NTHREAD];

int workk[NTHREAD];

using namespace std;

mutex locks[N_NODES];
typedef tuple<int, int, double> DataPoint;

struct Comp
{
  bool operator()(const pair<int, int>& s1, const pair<int, int>& s2) {
    return s1.second > s2.second;
  }
};

int myrandom (int i) { return std::rand()%i;}

void output_model() {
  for (int i = 0; i < N_NODES; i++) {
    cout << i << " ";
    for (int j = 0; j < K; j++) {
      cout << model[i][j] << " ";
    }
    cout << endl;
  }
}

void pin_to_core(size_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void update_coords() {
  for (int i = 0; i < N_NODES; i++) {
    double diff = N_DATAPOINTS - bookkeeping[i];
    for (int j = 0; j < K; j++) {
      model[i][j] -= GAMMA * diff * sum_gradients[i][j] / N_DATAPOINTS;
    }
  }
}

void clear_bookkeeping() {
  if (SAGA) {
    update_coords();
    for (int i = 0; i < N_NODES; i++) {
      bookkeeping[i] = 0;
    }
  }
}

double compute_loss(vector<DataPoint> points) {
  double loss = 0;
#pragma omp parallel for reduction(+:loss)
  for (int i = 0; i < points.size(); i++) {
    int u = get<0>(points[i]), v = get<1>(points[i]);
    double w = get<2>(points[i]);
    double sub_loss = 0;
    for (int j = 0; j < K; j++) {
      sub_loss += (model[u][j]+model[v][j]) *  (model[u][j]+model[v][j]);
      //sub_loss += (model[u][j]-model[v][j])* (model[u][j]-model[v][j]);
    }
    loss += w * (log(w) - sub_loss - C) * (log(w) - sub_loss - C);
    //loss += sub_loss;
  }
  return loss / points.size();
}

double compute_loss_for_record_epoch(vector<DataPoint> &points, int epoch) {
  double loss = 0;
#pragma omp parallel for reduction(+:loss)
  for (int i = 0; i < points.size(); i++) {
    int u = get<0>(points[i]), v = get<1>(points[i]);
    double w = get<2>(points[i]);
    double sub_loss = 0;
    for (int j = 0; j < K; j++) {
      sub_loss += (model_records[epoch][u][j]+model_records[epoch][v][j]) *  (model_records[epoch][u][j]+model_records[epoch][v][j]);
      //sub_loss += (model_records[epoch][u][j]-model_records[epoch][v][j])* (model_records[epoch][u][j]-model_records[epoch][v][j]);
    }
    loss += w * (log(w) - sub_loss - C) * (log(w) - sub_loss - C);    
    //loss += sub_loss;
  }
  return loss / (double)points.size();
}

double print_loss_for_records(vector<DataPoint> &p) {
  for (int i = 0; i < N_EPOCHS; i++) {
    double loss;
    loss = compute_loss_for_record_epoch(p, i);
    double overall_time = overall_times[i];
    double gradient_time = gradient_times[i];
    cout << loss << " " << overall_time << " " << gradient_time << endl;
  }
}

double copy_model_to_records(int epoch, double overall_time, double gradient_time) {
  gradient_times[epoch] = gradient_time;
  overall_times[epoch] = overall_time;
  for (int i = 0; i < N_NODES; i++) {
    for (int j = 0; j < K; j++) {
      model_records[epoch][i][j] = model[i][j];
    }
  }
}

void do_cyclades_gradient_descent_with_points(DataPoint * access_pattern, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, int thread_id, int epoch) {

  pin_to_core(thread_id);

  for (int batch = 0; batch < access_length.size(); batch++) {
    
    //Wait for all threads to be on the same batch
    if (SHOULD_SYNC) {
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
      int x = get<0>(p), y = get<1>(p);
      double r = get<2>(p);
      int indx = batch_index_start[batch]+i;
      int update_order = order[batch_index_start[batch]+i];      
      double diff_x = update_order - bookkeeping[x] - 1;
      double diff_y = update_order - bookkeeping[y] - 1;
      
      //if (diff_x <= 0) diff_x = 0;
      //if (diff_y <= 0) diff_y = 0;

      if (SAGA) {
	for (int j = 0; j < K; j++) {
	  model[x][j] -=  (double)GAMMA * diff_x * sum_gradients[x][j] / N_DATAPOINTS;
	  model[y][j] -=  (double)GAMMA * diff_y * sum_gradients[y][j] / N_DATAPOINTS;
	}
      }

      for (int j = 0; j < K; j++) {
	//cout << prev_grad[pt_index][x][j] << endl;
	//cout << sum_gradients[x][j] << endl;
	//cout << model[x][j] << endl;
	//cout << model[y][j] << endl;
	//cout << x << " " << y << endl;
      }

      //Get gradient multiplies
      double l2norm_sqr = 0;
      for (int j = 0; j < K; j++) {
	l2norm_sqr += (model[x][j] + model[y][j]) * (model[x][j] + model[y][j]);
      }
      double mult = 2 * r * (log(r) - l2norm_sqr - C);

      //Keep track of sums for optimizing C
      //C_sum_mult[thread_id][indx] = 2 * r * (log(r) - l2norm_sqr - C);
      C_sum_mult[thread_id][indx] = r * (log(r) - l2norm_sqr);
      C_sum_mult2[thread_id][indx] = r;

      //Apply gradient update
      for (int j = 0; j < K; j++) {
	double gradient =  -1 * (mult * 2 * (model[x][j] + model[y][j]));
	//double gradient = r * 2 * (model[x][j] - model[y][j]);
	
	if (SAGA) {
	  
	  /*model[x][j] -= GAMMA * (gradient - prevv_gradients[oorder[x][y]][j*2] + sum_gradients[x][j]) / min(n_datapoints_seen, N_DATAPOINTS);
	  sum_gradients[x][j] += gradient - prevv_gradients[oorder[x][y]][j*2];
	  prevv_gradients[oorder[x][y]][j*2] = gradient;

	  model[y][j] -= GAMMA * (gradient - prevv_gradients[oorder[x][y]][j*2+1] + sum_gradients[y][j]) / min(n_datapoints_seen, N_DATAPOINTS);
	  sum_gradients[y][j] += gradient - prevv_gradients[oorder[x][y]][j*2+1];
	  prevv_gradients[oorder[x][y]][j*2+1] = gradient;*/

	  if (epoch != 0) {
	    model[x][j] -= GAMMA * (gradient - prev_gradients[thread_id][indx][j*2] + sum_gradients[x][j] / N_DATAPOINTS);
	    //model[x][j] -= GAMMA * (gradient - prev_gradients[thread_id][indx][j*2] + sum_gradients[x][j]) / N_DATAPOINTS;
	    sum_gradients[x][j] += gradient - prev_gradients[thread_id][indx][j*2];
	    prev_gradients[thread_id][indx][j*2] = gradient;
	    
	    model[y][j] -= GAMMA * (gradient - prev_gradients[thread_id][indx][j*2+1] + sum_gradients[y][j] / N_DATAPOINTS);
	    //model[y][j] -= GAMMA * (gradient - prev_gradients[thread_id][indx][j*2+1] + sum_gradients[y][j]) / N_DATAPOINTS;
	    sum_gradients[y][j] += gradient - prev_gradients[thread_id][indx][j*2+1];
	    prev_gradients[thread_id][indx][j*2+1] = gradient;
	  }
	  else {
	    model[x][j] -= GAMMA * gradient;
	    model[y][j] -= GAMMA * gradient;
	    sum_gradients[x][j] += gradient - prev_gradients[thread_id][indx][j*2];
	    prev_gradients[thread_id][indx][j*2] = gradient;
	    sum_gradients[y][j] += gradient - prev_gradients[thread_id][indx][j*2+1];
	    prev_gradients[thread_id][indx][j*2+1] = gradient;
	  }
	  
	  /*model[x][j] -= GAMMA * (gradient - prev_gradients[thread_id][indx][j] + sum_gradients[x][j]) / min(n_datapoints_seen, N_DATAPOINTS);
	  model[y][j] -= GAMMA * (gradient - prev_gradients[thread_id][indx][j+K] + sum_gradients[y][j]) / min(n_datapoints_seen, N_DATAPOINTS);
	  sum_gradients[x][j] += gradient - prev_gradients[thread_id][indx][j];
	  sum_gradients[y][j] += gradient - prev_gradients[thread_id][indx][j+K];
	  prev_gradients[thread_id][indx][j] = gradient;
	  prev_gradients[thread_id][indx][j+K] = gradient;*/
	}
	else {
	  model[x][j] -= GAMMA * gradient;
	  model[y][j] -= GAMMA * gradient;
	  //model[x][j] -= GAMMA * gradient;
	  //model[y][j] -= GAMMA * gradient * -1;
	}
      }

      //Update bookkeeping
      bookkeeping[x] = update_order;
      bookkeeping[y] = update_order;
      //locks[x].unlock();
      //locks[y].unlock();
    }
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

  for (int i = 0; i < NTHREAD; i++) {
    //cout << total_size_needed[i] << " ";
  }
  //cout << endl;

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

void compute_CC_thread(map<int, vector<int> > &CCs, vector<DataPoint> &points, int start, int end, int thread_id) {
  pin_to_core(thread_id);
  //int tree[end-start + N_NODES];
  //int *tree =(int *) malloc(sizeof(int) * (end-start+N_NODES));  
  int *tree = tree_thread[thread_id];

  for (long long int i = 0; i < end-start + N_NODES; i++) 
    tree[i] = i;

  for (int i = start; i < end; i++) {
    DataPoint p = points[i];
    int src = i-start;
    int e1 = get<0>(p) + end-start;
    int e2 = get<1>(p) + end-start;
    int c1 = union_find(src, tree);
    int c2 = union_find(e1, tree);
    int c3 = union_find(e2, tree);
    tree[c3] = c1;
    tree[c2] = c1;
  }
  for (int i = 0; i < end-start; i++) {
    int group = union_find(i, tree);
    CCs[group].push_back(i+start);    
  }
  //free(tree);
 }

map<int, vector<int> > compute_CC(vector<DataPoint> &points, int start, int end) {
  //int tree[end-start + N_NODES];
  int *tree =(int *) malloc(sizeof(int) * (end-start+N_NODES));

  for (long long int i = 0; i < end-start + N_NODES; i++) 
    tree[i] = i;

  for (int i = start; i < end; i++) {
    DataPoint p = points[i];
    int src = i-start;
    int e1 = get<0>(p) + end-start;
    int e2 = get<1>(p) + end-start;
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
  free(tree);
  return CCs;
 }

vector<DataPoint> get_word_embeddings_data() {
  vector<DataPoint> datapoints;
  ifstream in(WORD_EMBEDDINGS_FILE);
  string s;
  while (getline(in, s)) {
    stringstream linestream(s);
    int n1, n2;
    double occ;
    linestream >> n1 >> n2 >> occ;
    datapoints.push_back(DataPoint(n1, n2, occ));
  }
  return datapoints;
}

void initialize_model() {
  for (int i = 0; i < N_NODES; i++) {
    for (int j = 0; j < K; j++) {
      model[i][j] = rand() / (double)RAND_MAX;
      //model[i][j] = .5;
    }
  }
}

void cyc_word_embeddings() {
  vector<DataPoint> points = get_word_embeddings_data();
  initialize_model();
  random_shuffle(points.begin(), points.end());
  Timer overall;

  //Access pattern generation
  int n_batches = (int)ceil((points.size() / (double)BATCH_SIZE));

  //Access pattern of form [thread][batch][point_ids]
  vector<DataPoint *> access_pattern(NTHREAD);
  //Access length (number of elements in corresponding cc)
  vector<vector<int > > access_length(NTHREAD);
  vector<vector<int > > batch_index_start(NTHREAD);
  vector<vector<int > > order(NTHREAD);
  
  for (int i = 0; i < NTHREAD; i++) {
    access_length[i].resize(n_batches);
    batch_index_start[i].resize(n_batches);
    order[i].resize(n_batches);
  }

  //CC Generation serial
  /*Timer t2;
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
    }*/
  //CC Parallel
  map<int, vector<int> > CCs[n_batches];
#pragma omp parallel for
  for (int i = 0; i < n_batches; i++) {
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());
    compute_CC_thread(CCs[i], points, start, end, omp_get_thread_num());
  }
  Timer allocation;
  for (int i = 0; i < n_batches; i++) {
    distribute_ccs(CCs[i], access_pattern, access_length, batch_index_start, i, points, order);
  }

  for (int i = 0; i < NTHREAD; i++) {
    //prev_gradients[i] = (double **)malloc(sizeof(double *) * thread_load_balance[i]);
    C_sum_mult[i] = (double *)malloc(sizeof(double) * thread_load_balance[i]);
    C_sum_mult2[i] = (double *)malloc(sizeof(double) * thread_load_balance[i]);
    for (int j = 0; j < thread_load_balance[i]; j++) {
      C_sum_mult[i][j] = 0;
      C_sum_mult2[i][j] = 0;
      //prev_gradients[i][j] = (double *)malloc(sizeof(double) * 2*K_TO_CACHELINE);
      for (int k = 0; k < 2*K_TO_CACHELINE; k++) {
	//prev_gradients[i][j][k] = 0;
      }
    }
  }

  //Perform cyclades
  float copy_time = 0;
  Timer gradient_time;
  for (int i = 0; i < N_EPOCHS; i++) {
    //cout << compute_loss(points) << endl;
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
      Timer copy_timer;
      //copy_model_to_records(i, overall.elapsed()-copy_time, gradient_time.elapsed()-copy_time);
      cout << compute_loss(points) << " " << overall.elapsed()-copy_time << " " << gradient_time.elapsed()-copy_time << endl;
      copy_time += copy_timer.elapsed();
    }
    vector<thread> threads;
    if (NTHREAD == 1) {
      do_cyclades_gradient_descent_with_points(access_pattern[0], access_length[0], batch_index_start[0], order[0], 0, i);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, ref(access_pattern[j]), ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j, i));
      }
      for (int j = 0; j < threads.size(); j++) {
	threads[j].join();
      }
      for (int j = 0; j < NTHREAD; j++) {
	thread_batch_on[j] = 0;
      }
    }
    clear_bookkeeping();

    //Optimize C
    double C_A = 0, C_B = 0;
    #pragma omp parallel for reduction(+:C_A,C_B)
    for (int t = 0; t < NTHREAD; t++) {
      for (int d = 0; d < order[t].size(); d++) {
	C_A += C_sum_mult[t][d];
	C_B += C_sum_mult2[t][d];
      }
    }
    C = C_A / C_B;
    GAMMA *= GAMMA_REDUCTION;
  }
  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
  else {
    //print_loss_for_records(points);
  }
}

void hog_word_embeddings() {
  vector<DataPoint> points = get_word_embeddings_data();
  initialize_model();
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
    order[i].resize(1);

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

  for (int i = 0; i < NTHREAD; i++) {
    //prev_gradients[i] = (double **)malloc(sizeof(double *) * order[i].size());
    C_sum_mult[i] = (double *)malloc(sizeof(double) * order[i].size());
    C_sum_mult2[i] = (double *)malloc(sizeof(double) * order[i].size());
    for (int j = 0; j < order[i].size(); j++) {
      C_sum_mult[i][j] = 0;
      C_sum_mult2[i][j] = 0;
      //prev_gradients[i][j] = (double *)malloc(sizeof(double) * 2*K_TO_CACHELINE);
      for (int k = 0; k < 2*K_TO_CACHELINE; k++) {
	//prev_gradients[i][j][k] = 0;
      }
    }
  }

  //Divide to threads
  float copy_time = 0;
  Timer gradient_time;
  for (int i = 0; i < N_EPOCHS; i++) {
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
      Timer copy_timer;
      //  copy_model_to_records(i, overall.elapsed()-copy_time, gradient_time.elapsed()-copy_time);
      cout << compute_loss(points) << " " << overall.elapsed()-copy_time << " " << gradient_time.elapsed()-copy_time << endl;
      copy_time += copy_timer.elapsed();
    }
    //cout << compute_loss(points) << endl;
    vector<thread> threads;
    if (NTHREAD == 1) {
      do_cyclades_gradient_descent_with_points(access_pattern[0], access_length[0], batch_index_start[0], order[0], 0, i);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, ref(access_pattern[j]), ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j, i));
      }
      for (int j = 0; j < threads.size(); j++) {
	threads[j].join();
      }
    }
    clear_bookkeeping();
    
    //Optimize C
    //Optimize C
    double C_A = 0, C_B = 0;
#pragma omp parallel for reduction(+:C_A,C_B)
    for (int t = 0; t < NTHREAD; t++) {
      for (int d = 0; d < order[t].size(); d++) {
	C_A += C_sum_mult[t][d];
	C_B += C_sum_mult2[t][d];
      }
    }
    C = C_A / C_B;

    GAMMA *= GAMMA_REDUCTION;
  }

  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
  else {
    //print_loss_for_records(points);
  }
}

int main(void) {
  omp_set_num_threads(NTHREAD);  
  srand(0);
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(10);
  pin_to_core(0);

  //Allocate memory for tree for CC per thread
  for (int i = 0; i < NTHREAD; i++) {
    tree_thread[i] = (int *)malloc(sizeof(int) * (BATCH_SIZE + N_NODES));
  }

  //sum_gradients = (double **)malloc(sizeof(double *) * N_NODES);
  model = (double **)malloc(sizeof(double *) * N_NODES);
  for (int i = 0; i < N_NODES; i++) {
    //sum_gradients[i] = (double *)malloc(sizeof(double) * K_TO_CACHELINE);
    model[i] = (double *)malloc(sizeof(double) * K_TO_CACHELINE);
  }

  for (int i = 0; i < N_NODES; i++) {
    bookkeeping[i] = 0;
    for (int j = 0; j < K_TO_CACHELINE; j++) {
      //sum_gradients[i][j] = 0;
    }
  }

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
    thread_batch_on[i] = 0;
    workk[i] = 0;
  }

  if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    /*for (int i = 0; i < N_EPOCHS; i++) {
      model_records[i] = (double **)malloc(sizeof(double *) * N_NODES);
      for (int j = 0; j < N_NODES; j++) {
	model_records[i][j] = (double *)malloc(sizeof(double ) * K);
	
      }
      }*/
  }

  if (HOG) {
    hog_word_embeddings();
  }
  if (CYC) {
    cyc_word_embeddings();
  }
  //if (OUTPUT_MODEL) {
  //output_model();
    //}
}
