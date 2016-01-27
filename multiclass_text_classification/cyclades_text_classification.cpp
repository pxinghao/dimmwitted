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

/*#define TEXT_CLASSIFICATION_FILE "random_lines_filtered"
#define N_COORDS 411896
#define N_CATEGORIES 2
#define N_CATEGORIES_CACHE_ALIGNED (N_CATEGORIES / 8 + 1) * 8
#define N_DATAPOINT 120132*/

#define TEXT_CLASSIFICATION_FILE "random_lines_filtered_filtered"
#define N_COORDS 819
#define N_CATEGORIES 2
#define N_CATEGORIES_CACHE_ALIGNED (N_CATEGORIES / 8 + 1) * 8
#define N_DATAPOINT 100


#ifndef NTHREAD
#define NTHREAD 1
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 10
#endif 

#ifndef BATCH_SIZE
//#define BATCH_SIZE 400
#define BATCH_SIZE 1
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

#ifndef SAGA
#define SAGA 1
#endif

#ifndef START_GAMMA
#define START_GAMMA 1e-3 // SAGA
//#define START_GAMMA 2 // SGD
#endif

double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

int volatile thread_batch_on[NTHREAD];

//double ** prev_gradients[NTHREAD];
double prev_gradients[N_DATAPOINT][N_COORDS][N_CATEGORIES];
double ** model, **sum_gradients;
int bookkeeping[N_COORDS];

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
  //for (int j = 0; j < N_CATEGORIES; j++) 
  //probs[j] = 0;
  memset(probs, 0, sizeof(double) * N_CATEGORIES);
  
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

void do_cyclades_gradient_descent_with_points(DataPoint *access_pattern, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, int thread_id, int epoch) {
  pin_to_core(thread_id);

  double probs[N_CATEGORIES];
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

      DataPoint p = access_pattern[batch_index_start[batch]+i];
      int chosen_category = p.first;
      vector<pair<int, double> > sparse_array = p.second;
      int indx = batch_index_start[batch]+i;
      int update_order = order[indx];

      if (SAGA) {
	for (int k = 0; k < sparse_array.size(); k++) {
	  for (int j = 0; j < N_CATEGORIES; j++) {
	    double diff = update_order - bookkeeping[sparse_array[k].first] - 1; 
	    model[sparse_array[k].first][j] -= GAMMA * sum_gradients[sparse_array[k].first][j] * diff / N_DATAPOINT;
	  }
	}
      }

      
      for (int i = 0; i < sparse_array.size(); i++) {
	for (int j = 0; j < N_CATEGORIES; j++) {
	  //cout << prev_gradients[pt_index][first_coord][j] << endl;
	  //cout << sum_gradients[first_coord][j] << endl;
	  //cout << sparse_array[i].first << endl;
	  //cout << model[sparse_array[i].first][j] << endl;
	  //cout << model[second_coord][j] << endl;
	}
      }

      //Clear probs
      compute_probs(p, probs);
      
      //Do gradient descent
      for (int j = 0; j < sparse_array.size(); j++) {
	for (int k = 0; k < N_CATEGORIES; k++) {
	  int is_correct = k == chosen_category ? 1 : 0;
	  double gradient = (-sparse_array[j].second * (is_correct - probs[k]));
	  if (SAGA) {
	    /*model[sparse_array[j].first][k] -= GAMMA * (gradient - prev_gradients[thread_id][indx][j+k] + 
							sum_gradients[sparse_array[j].first][k] / N_DATAPOINT);
	    sum_gradients[sparse_array[j].first][k] += gradient - prev_gradients[thread_id][indx][j+k];
	    prev_gradients[thread_id][indx][j+k] = gradient;*/

	    model[sparse_array[j].first][k] -= GAMMA * (gradient - prev_gradients[update_order-1][sparse_array[j].first][k] + 
							sum_gradients[sparse_array[j].first][k] / N_DATAPOINT);
	    sum_gradients[sparse_array[j].first][k] += gradient -  prev_gradients[update_order-1][sparse_array[j].first][k];
	    prev_gradients[update_order-1][sparse_array[j].first][k] = gradient;
	  }
	  else {
	    model[sparse_array[j].first][k] -= GAMMA * gradient;
	  }
	}
	bookkeeping[sparse_array[j].first] = update_order;
      }
    }
  }
} 

void distribute_ccs(map<int, vector<int> > &ccs, vector<vector<DataPoint> > &access_pattern, vector<vector<int> > &access_length, vector<vector<int> > &batch_index_start, int batchnum, vector<DataPoint> &points, vector<vector<int> > &order) {
  
  int * chosen_threads = (int *)malloc(sizeof(int) * ccs.size());
  int total_size_needed[NTHREAD];
  int count = 0;
  vector<pair<int, int> > balances;

  for (int i = 0; i < NTHREAD; i++) {
    total_size_needed[i] = 0;
    balances.push_back(pair<int, int>(i, 0));
  }

  make_heap(balances.begin(), balances.end(), Comp());

  //Count total size needed for each access pattern
  double max_cc_size = 0, avg_cc_size = 0;
  for (map<int, vector<int> >::iterator it = ccs.begin(); it != ccs.end(); it++, count++) {
    pair<int, int> best_balance = balances.front();
    int chosen_thread = best_balance.first;
    chosen_threads[count] = chosen_thread;
    pop_heap(balances.begin(), balances.end(), Comp()); balances.pop_back();
    vector<int> cc = it->second;
    total_size_needed[chosen_thread] += cc.size();
    
    //best_balance.second += cc.size();
    int num_coords = 0;
    for (int i = 0; i < cc.size(); i++) num_coords += points[cc[i]].second.size();
    best_balance.second += num_coords;
    
    balances.push_back(best_balance); push_heap(balances.begin(), balances.end(), Comp());
    max_cc_size = max(max_cc_size, (double) cc.size());
    avg_cc_size += cc.size();
  }
  avg_cc_size /= (double)ccs.size();

  for (int i = 0; i < balances.size(); i++) {
    //cout << balances[i].second << " ";
  }
  //cout << endl;
  
  //Allocate memory
  int index_count[NTHREAD];
  int max_load = 0;
  for (int i = 0; i < NTHREAD; i++) {

    size_t new_size = (size_t)total_size_needed[i] + cur_datapoints_used[i];
    batch_index_start[i][batchnum] = cur_datapoints_used[i];
    access_pattern[i].resize(new_size);
    cur_datapoints_used[i] += total_size_needed[i];
    order[i].resize(cur_datapoints_used[i]);
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
    double label;
    linestream >> label;
    string coord_freq_str, coord_str, freq_str;
    vector<pair<int, double> > coord_freq_pairs;
    int num_coords_encountered = 0;
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
    if (label_map.find((int)label) == label_map.end()) 
      label_map[(int)label] = labels.size();

    datapoints.push_back(DataPoint(label_map[(int)label], coord_freq_pairs));
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

void update_coords() {
  for (int i = 0; i < N_COORDS; i++) {
    double diff = N_DATAPOINT - bookkeeping[i];
    for (int j = 0; j < N_CATEGORIES; j++) {
      model[i][j] -= GAMMA * diff * sum_gradients[i][j] / N_DATAPOINT;
    }
  }
}

void clear_bookkeeping() {
  if (SAGA) {
    update_coords();
    for (int i = 0; i < N_COORDS; i++) {
      bookkeeping[i] = 0;
    }
  }
}

void cyc_text_classification() {
  vector<DataPoint> points = get_text_classification_data();

  initialize_model();
  //random_shuffle(points.begin(), points.end());

  Timer overall;

  //Access pattern generation
  int n_batches = (int)ceil((points.size() / (double)BATCH_SIZE));

  //Access pattern of form [thread][batch][point_ids]
  vector<vector<DataPoint> > access_pattern(NTHREAD);
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
  //#pragma omp parallel for
  for (int i = 0; i < n_batches; i++) {
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());
    compute_CC_thread(CCs[i], points, start, end, omp_get_thread_num());
  }

  for (int i = 0; i < n_batches; i++) {
    distribute_ccs(CCs[i], access_pattern, access_length, batch_index_start, i, points, order);
  }
  //exit(0);

  /*for (int i = 0; i < NTHREAD; i++) {
    prev_gradients[i] = (double **)malloc(sizeof(double *) * thread_load_balance[i]);
    for (int j = 0; j < thread_load_balance[i]; j++) {
      int size = access_pattern[i][j].second.size();
      prev_gradients[i][j] = (double *)malloc(sizeof(double) * size * N_CATEGORIES);
      for (int k = 0; k < size * N_CATEGORIES; k++) {
	prev_gradients[i][j][k] = 0;
      }
    }
    }*/

  //Perform cyclades
  float copy_time = 0;
  Timer gradient_time;
  for (int i = 0; i < N_EPOCHS; i++) {
    if (SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
      Timer copy_timer;
      cout << compute_loss(points) << " " << overall.elapsed()-copy_time << " " << gradient_time.elapsed()-copy_time << endl;
      copy_time += copy_timer.elapsed();
    }
    vector<thread> threads;
    if (NTHREAD == 1) {
      do_cyclades_gradient_descent_with_points((DataPoint *)&access_pattern[0][0], access_length[0], batch_index_start[0], order[0], 0, i);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, (DataPoint *)&access_pattern[j][0], ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j, i));
      }
      for (int j = 0; j < threads.size(); j++) {
	threads[j].join();
      }
      for (int j = 0; j < NTHREAD; j++) {
	thread_batch_on[j] = 0;
      }
    }
    GAMMA *= GAMMA_REDUCTION;    
    clear_bookkeeping();
  }
  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
}

void hog_text_classification() {
  vector<DataPoint> points = get_text_classification_data();
  initialize_model();
  random_shuffle(points.begin(), points.end());
  Timer overall;

  //Hogwild access pattern construction
  vector<vector<DataPoint> > access_pattern(NTHREAD);
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
    access_pattern[i].resize(n_points_per_thread);
    order[i].resize(n_points_per_thread);
    for (int j = start; j < end; j++) {
      access_pattern[i][j-start] = points[j];
      order[i][j-start] = j+1;
    }
    access_length[i][0] = n_points_per_thread;
  }

  /*for (int i = 0; i < NTHREAD; i++) {
    prev_gradients[i] = (double **)malloc(sizeof(double *) * order[i].size());
    for (int j = 0; j < order[i].size(); j++) {
      int size = access_pattern[i][j].second.size();
      prev_gradients[i][j] = (double *)malloc(sizeof(double) * size * N_CATEGORIES);
      for (int k = 0; k < size * N_CATEGORIES; k++) {
	prev_gradients[i][j][k] = 0;
      }
    }
    }*/

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
      do_cyclades_gradient_descent_with_points((DataPoint *)&access_pattern[0][0], access_length[0], batch_index_start[0], order[0], 0, i);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, (DataPoint *)&access_pattern[j][0], ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j, i));
      }
      for (int j = 0; j < threads.size(); j++) {
	threads[j].join();
      }
    }
    GAMMA *= GAMMA_REDUCTION;
    clear_bookkeeping();
  }
  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
}

int main(void) {
  omp_set_num_threads(NTHREAD);  
  srand(0);
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(20);
  pin_to_core(0);

  sum_gradients = (double **)malloc(sizeof(double *) * N_COORDS);
  model = (double **)malloc(sizeof(double *) * N_COORDS);
  for (int i = 0; i < N_COORDS; i++) {
    model[i] = (double *)malloc(sizeof(double) * N_CATEGORIES_CACHE_ALIGNED);
    sum_gradients[i] = (double *)malloc(sizeof(double) * N_CATEGORIES_CACHE_ALIGNED);
  }

  for (int i = 0; i < N_COORDS; i++) {
    bookkeeping[i] = 0;
    for (int j = 0; j < N_CATEGORIES_CACHE_ALIGNED; j++) {
      sum_gradients[i][j] = 0;
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
  for (int i = 0; i < NTHREAD; i++)
    cout << thread_load_balance[i] << endl;
}
