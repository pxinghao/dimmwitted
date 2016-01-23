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
#include <cmath>

#define TEXT_CLASSIFICATION_FILE "dry-run_lshtc_dataset/Task1_Train:CrawlData_Test:CrawlData/train.txt"
#define N_COORDS 51033
#define N_CATEGORIES 1139
#define N_DATAPOINTS 4463

#ifndef NTHREAD
#define NTHREAD 8
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 50
#endif 

#ifndef BATCH_SIZE
#define BATCH_SIZE 1000 //full 80 mb
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
#undef SHOULD_SYNC
#define SHOULD_SYNC 0
#endif
#ifndef SHOULD_SYNC
#define SHOULD_SYNC 1
#endif

#ifndef SAG
#define SAG 0
#endif

#ifndef START_GAMMA
#define START_GAMMA 2e-11 // SGD
#endif

double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

int volatile thread_batch_on[NTHREAD];

double ** prev_gradients[NTHREAD];
double ** model;

double thread_load_balance[NTHREAD];
size_t cur_bytes_allocated[NTHREAD];
int cur_datapoints_used[NTHREAD];
int core_to_node[NTHREAD];

using namespace std;

typedef pair<int, vector<pair<int, double> > >  DataPoint;

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

void compute_probs(DataPoint &p, double *probs) {

  double sum_prob = 0;
  vector<pair<int, double> > sparse_array = p.second;

  //Clear probs
  for (int j = 0; j < N_CATEGORIES; j++) 
    probs[j] = 0;
  
  //x^t A_i
  for (int j = 0; j < sparse_array.size(); j++) {
    for (int k = 0; k < N_CATEGORIES; k++) {
      probs[k] += model[sparse_array[j].first][k] * sparse_array[j].second;
    }
  }
  for (int j = 0; j < N_CATEGORIES; j++) {
    probs[j] = exp(probs[j]);
    sum_prob += probs[j];
  }
  
  //Normalize
  for (int j = 0; j < N_CATEGORIES; j++) {
    probs[j] /= sum_prob;
  }
}

double compute_loss(vector<DataPoint> &points) {
  double loss = 0;
  double probs[N_CATEGORIES];
  for (int m = 0; m < points.size(); m++) {
    compute_probs(points[m], probs);
    loss += log(probs[points[m].first]);
  }
  return -loss / points.size();
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
      int chosen_category = p.first;
      vector<pair<int, double> > sparse_array = p.second;
      int indx = batch_index_start[batch]+i;
      int update_order = order[indx];
      
      double probs[N_CATEGORIES];

      //compute probabilities
      compute_probs(p, probs);

      //Do gradient descent
      for (int j = 0; j < sparse_array.size(); j++) {
	for (int k = 0; k < N_CATEGORIES; k++) {
	  int is_correct = (k == chosen_category) ? 1 : 0;
	  model[j][k] -= GAMMA * (-sparse_array[j].second * (is_correct - probs[k]));
	}
      }
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
	
	access_pattern[i] = (DataPoint *)numa_realloc(access_pattern[i], old_size, new_size);
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
  int tree[end-start + N_COORDS];

  for (long long int i = 0; i < end-start + N_COORDS; i++) 
    tree[i] = i;

  for (int i = start; i < end; i++) {
    DataPoint p = points[i];
    vector<pair<int, double> > touched_coords = get<1>(p);
    int src = i-start;
    int src_group = union_find(src, tree);
    for (int k = 0; k < touched_coords.size(); k++) {
      int element = union_find(touched_coords[k].first + end-start, tree);
      tree[element] = src_group;
    }
  }
  for (int i = 0; i < end-start; i++) {
    int group = union_find(i, tree);
    CCs[group].push_back(i+start);    
  }
}

vector<DataPoint> get_text_classification_data() {
  vector<DataPoint> datapoints;
  ifstream in(TEXT_CLASSIFICATION_FILE);
  string s;
  int max_label = 0;
  int max_coord = 0;
  set<int> labels, coords;
  map<int, int> label_map, coord_map;
  while (getline(in, s)) {
    if (s.size() == 0) continue;
    stringstream linestream(s);
    int label;
    linestream >> label;
    string coord_freq_str, coord_str, freq_str;
    vector<pair<int, double> > coord_freq_pairs;
    while (linestream >> coord_freq_str) {
      stringstream coord_freq_stream(coord_freq_str);
      getline(coord_freq_stream, coord_str, ':');
      getline(coord_freq_stream, freq_str, ':');
      int coord;
      double freq;
      coord = stoi(coord_str);
      freq = stod(freq_str);

      //Remap coordinates to prevent gaps
      if (coord_map.find(coord) == coord_map.end()) 
	coord_map[coord] = coords.size();
      
      coord_freq_pairs.push_back(pair<int, double>(coord_map[coord], freq));
      coords.insert(coord);
    }

    //Remap labesl to prevent gaps
    if (label_map.find(label) == label_map.end()) 
      label_map[label] = labels.size();

    datapoints.push_back(DataPoint(label_map[label], coord_freq_pairs));
    labels.insert(label);
  }

  cout << "NUMBER OF COORDS: " << coords.size() << endl;
  cout << "NUMBER OF LABELS: " << labels.size() << endl; 
  cout << "NUMBER OF DATAPOINTS: " << datapoints.size() << endl;
  return datapoints;
}

void initialize_model() {
  for (int j = 0; j < N_COORDS; j++) {
    for (int i = 0; i < N_CATEGORIES; i++) {      
      model[j][i] = 0;
    }
  }
}

void cyc_text_classification() {
  vector<DataPoint> points = get_text_classification_data();

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

  //CC Parallel
  map<int, vector<int> > CCs[n_batches];
#pragma omp parallel for
  for (int i = 0; i < n_batches; i++) {
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());
    compute_CC_thread(CCs[i], points, start, end, omp_get_thread_num());
  }
  for (int i = 0; i < n_batches; i++) {
    distribute_ccs(CCs[i], access_pattern, access_length, batch_index_start, i, points, order);
  }

  //Perform cyclades
  float copy_time = 0;
  Timer gradient_time;
  for (int i = 0; i < N_EPOCHS; i++) {
    cout << compute_loss(points) << endl;
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
      Timer copy_timer;
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
    GAMMA *= GAMMA_REDUCTION;    
  }
  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
}

void hog_text_classification() {
  /*vector<DataPoint> points = get_text_classification_data();
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
    prev_gradients[i] = (double **)malloc(sizeof(double *) * order[i].size());
    C_sum_mult[i] = (double *)malloc(sizeof(double) * order[i].size());
    for (int j = 0; j < order[i].size(); j++) {
      C_sum_mult[i][j] = 0;
      prev_gradients[i][j] = (double *)malloc(sizeof(double) * 2*K_TO_CACHELINE);
      for (int k = 0; k < 2*K_TO_CACHELINE; k++) {
	prev_gradients[i][j][k] = 0;
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
    
    double c_gradient = 0;
#pragma omp parallel for reduction(+:c_gradient)
    for (int t = 0; t < NTHREAD; t++) {
      for (int d = 0; d < order[t].size(); d++) {
	c_gradient += -C_sum_mult[t][d];
      }
    }
    C -= GAMMA * c_gradient;

    GAMMA *= GAMMA_REDUCTION;
  }

  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
  else {
    //print_loss_for_records(points);
    }*/
}

int main(void) {
  omp_set_num_threads(NTHREAD);  
  srand(0);
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(20);
  pin_to_core(0);

  model = (double **)malloc(sizeof(double *) * N_COORDS);
  for (int i = 0; i < N_COORDS; i++) {
    model[i] = (double *)malloc(sizeof(double) * N_CATEGORIES);
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

  //Clear miscellaneous datastructures for CC load balancing 
  for (int i = 0; i < NTHREAD; i++) {
    cur_bytes_allocated[i] = 0;
    cur_datapoints_used[i] = 0;
    thread_batch_on[i] = 0;
  }

  if (HOG) {
    hog_text_classification();
  }
  if (CYC) {
    cyc_text_classification();
  }
}
