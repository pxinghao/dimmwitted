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

#define WORD_EMBEDDINGS_FILE "sparse_graph"
//#define N_NODES 213271
//#define N_DATAPOINTS 20207156
#define N_NODES 3822
#define N_DATAPOINTS 80821

#define NTHREAD 1

#ifndef N_EPOCHS
#define N_EPOCHS 10
#endif 

#ifndef BATCH_SIZE
#define BATCH_SIZE 200
#endif

#ifndef HOG
#define HOG 0
#endif

#ifndef CYC
#define CYC 0
#endif

#ifndef SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH
#define SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH 0
#endif

#if HOG == 1
#define SHOULD_SYNC 0
#endif
#ifndef SHOULD_SYNC
#define SHOULD_SYNC 1
#endif

//k way cuts
#define K 2
#define K_TO_CACHELINE ((K / 8 + 1) * 8)

#ifndef SAG
#define SAG 1
#endif

#ifndef START_GAMMA 
#define START_GAMMA .1
#endif

double C = 0;
double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

int volatile thread_batch_on[NTHREAD];

double avg_gradients[N_NODES][K], prev_gradients[N_NODES][K];
double model[N_NODES][K_TO_CACHELINE] __attribute__((aligned(64)));
double **model_records[N_EPOCHS];
int volatile bookkeeping[N_NODES];

double gradient_times[N_EPOCHS], overall_times[N_EPOCHS];
double thread_load_balance[NTHREAD];
size_t cur_bytes_allocated[NTHREAD];
int cur_datapoints_used[NTHREAD];
int core_to_node[NTHREAD];

int workk[NTHREAD];

using namespace std;
typedef tuple<int, int, double> DataPoint;

struct Comp
{
  bool operator()(const pair<int, int>& s1, const pair<int, int>& s2) {
    return s1.second > s2.second;
  }
};

void pin_to_core(size_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void clear_bookkeeping() {
  for (int i = 0; i < N_NODES; i++) {
    bookkeeping[i] = 0;
    for (int j = 0; j < K; j++) {
      avg_gradients[i][j] = prev_gradients[i][j] = 0;
    }
  }
}

double compute_loss(vector<DataPoint> points) {
  double loss = 0;
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

void do_cyclades_gradient_descent_with_points(DataPoint * access_pattern, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, int thread_id) {
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
      int update_order = order[batch_index_start[batch]+i];      
      double diff_x = update_order - bookkeeping[x] - 1;
      double diff_y = update_order - bookkeeping[y] - 1;

      if (diff_x <= 0) diff_x = 0;
      if (diff_y <= 0) diff_y = 0;

      if (SAG) {
	for (int j = 0; j < K; j++) {
	  model[x][j] -=  (double)GAMMA * diff_x * avg_gradients[x][j] / N_DATAPOINTS;
	  model[y][j] -=  (double)GAMMA * diff_y * avg_gradients[y][j] / N_DATAPOINTS;
	}
      }

      //Get gradient multiplies
      double l2norm_sqr = 0;
      for (int j = 0; j < K; j++) {
	l2norm_sqr += (model[x][j] + model[y][j]) * (model[x][j] + model[y][j]);
      }
      double mult = 2 * r * (log(r) - l2norm_sqr - C);

      //Apply gradient update
      for (int j = 0; j < K; j++) {
	double gradient = -1 * (mult * 2 * (model[x][j] + model[y][j]));
	
	if (SAG) {
	  model[x][j] -= GAMMA * (gradient - prev_gradients[x][j] + avg_gradients[x][j]) / N_DATAPOINTS;
	  model[y][j] -= GAMMA * (gradient - prev_gradients[y][j] + avg_gradients[y][j]) / N_DATAPOINTS;
	  avg_gradients[x][j] += (gradient - prev_gradients[x][j]);
	  avg_gradients[y][j] += (gradient - prev_gradients[y][j]);
	  prev_gradients[y][j] = gradient;
	  prev_gradients[x][j] = gradient;
	}
	else {
	  model[x][j] -= GAMMA * gradient;
	  model[y][j] -= GAMMA * gradient;
	}
      }

      //Update bookkeeping
      bookkeeping[x] = update_order;
      bookkeeping[y] = update_order;
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
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) copy_model_to_records(i, overall.elapsed(), gradient_time.elapsed());
    clear_bookkeeping();
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
    GAMMA *= GAMMA_REDUCTION;


    //START
    /*if (i % 1 == 0) {
      random_shuffle(points.begin(), points.end());
      for (int ii = 0; ii < NTHREAD; ii++) {
	thread_load_balance[ii] = 0;
	cur_datapoints_used[ii] = 0;
	}
	for (int ii = 0; ii < n_batches; ii++) {
	
	int start = ii * BATCH_SIZE;
	int end = min((ii+1)*BATCH_SIZE, (int)points.size());
	
	///Compute connected components of data points
	Timer ttt;
	map<int, vector<int> > cc = compute_CC(points, start, end);
	cc_time += ttt.elapsed();
	//Distribute connected components across threads
	Timer ttt2;
	distribute_ccs(cc, access_pattern, access_length, batch_index_start, ii, points);
	alloc_time += ttt2.elapsed();
	}
	       
	//END
	}*/
  }

  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
  else {
    print_loss_for_records(points);
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

  //Divide to threads
  Timer gradient_time;

  for (int i = 0; i < N_EPOCHS; i++) {
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) copy_model_to_records(i, overall.elapsed(), gradient_time.elapsed());
    //cout << compute_loss(points) << endl;
    clear_bookkeeping();
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
    GAMMA *= GAMMA_REDUCTION;
  }

  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
  else {
    print_loss_for_records(points);
  }
}

int main(void) {
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(100);
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
    thread_batch_on[i] = 0;
    workk[i] = 0;
  }
  
  for (int i = 0; i < N_NODES; i++) {
    for (int j = 0; j < K; j++) {
      avg_gradients[i][j] = 0;
      prev_gradients[i][j] = 0;
    }
  }

  if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    for (int i = 0; i < N_EPOCHS; i++) {
      model_records[i] = (double **)malloc(sizeof(double **) * N_NODES);
      for (int j = 0; j < N_NODES; j++) {
	model_records[i][j] = (double *)malloc(sizeof(double *) * K);
      }
    }
  }

  if (HOG) {
    hog_word_embeddings();
  }
  if (CYC) {
    cyc_word_embeddings();
  }
}
