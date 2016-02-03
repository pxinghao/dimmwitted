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

//#define MATRIX_DATA_FILE "../SVRG/roadNet-CA.txt"
//#define N_DIMENSION 1965207
//#define MATRIX_DATA_FILE "delaunay_n13/delaunay_n13.mtx"
//#define N_DIMENSION 8193
//#define MATRIX_DATA_FILE "delaunay_n11/delaunay_n11.mtx"
//#define N_DIMENSION 2049
//#define MATRIX_DATA_FILE "../SVRG/nh2010/nh2010.mtx"
//#define N_DIMENSION 48838
//#define VERTEX_WEIGHT_FILE "../SVRG/nh2010/nh2010_population.mtx"
#define MATRIX_DATA_FILE "../SVRG/dblp-author/out.dblp-author"
#define N_DIMENSION 5425964
//#define MATRIX_DATA_FILE "ego-gplus/out.ego-gplus"
//#define N_DIMENSION 23629

//#define MATRIX_DATA_FILE "youtube-u-growth/out.youtube-u-growth"
//#define N_DIMENSION 3223589+1
#define N_DIMENSION_CACHE_ALIGNED (N_DIMENSION/8+1) * 8
#define NUM_SPARSE_ELEMENTS_IN_ROW 2

//#define N_DIMENSION 10000



#ifndef NTHREAD
#define NTHREAD 4
#endif

#define RANGE 100

#ifndef N_EPOCHS
#define N_EPOCHS 100
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 1000 //nh2010
//#define BATCH_SIZE 10000
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

#ifndef START_GAMMA
#define START_GAMMA -1
#endif

#ifndef CRIMP
#define CRIMP 1
#endif

#ifndef SET_GAMMA
//NH2010
//#define SET_GAMMA 3e-14 //BEST SAGA CYC NH2010
//#define SET_GAMMA 1e-14 //BEST SAGA HOG NH2010 32 thread
//#define SET_GAMMA 1e-14 //BEST SAGA HOG NH2010 16 thread
//#define SET_GAMMA 1e-14 //BEST SAGA HOG NH2010 8 thread
//#define SET_GAMMA 1e-14 //BEST SAGA HOG NH2010 4 thread
//#define SET_GAMMA 3e-14 //BEST SAGA HOG NH2010 1 thread

//#define SET_GAMMA 3e-4 //CYC DBLP
#define SET_GAMMA 1e-5 //HOG DBLP 32 thread
#define SET_GAMMA 1e-5 //HOG DBLP 16 thread
#define SET_GAMMA 1e-5 //HOG DBLP 8 thread
#define SET_GAMMA 1e-5 //HOG DBLP 4 thread
#define SET_GAMMA 3e-4 //HOG DBLP 1 thread
#endif

#ifndef SAGA
#define SAGA 1
#endif

using namespace std;

double GAMMA = START_GAMMA < 0 ? SET_GAMMA : START_GAMMA;
double GAMMA_REDUCTION = 1;

int volatile thread_batch_on[NTHREAD];

//Problem: minimize datapoints * model - B

int *trees[NTHREAD];
map<int, double> prev_gradients[N_DIMENSION];
double sum_gradients[N_DIMENSION];
double model[N_DIMENSION][1] __attribute__((aligned(64)));
double B[N_DIMENSION];

int bookkeeping[N_DIMENSION];

double thread_load_balance[NTHREAD];
size_t cur_bytes_allocated[NTHREAD];
int cur_datapoints_used[NTHREAD];
int core_to_node[NTHREAD];

int remainder_batch_size = 0;

//row #, sparse_vector of row
typedef pair<int, map<int, double> > DataPoint;

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


void mat_vect_mult(vector<DataPoint> &sparse_matrix, double *in, double *out) {
  memset(out, 0, sizeof(double) * N_DIMENSION);
  for (int j = 0; j < sparse_matrix.size(); j++) {
    map<int, double> sparse_row = sparse_matrix[j].second;
    int row = sparse_matrix[j].first;

    for (auto const & x : sparse_row) {
      out[row] += in[x.first] * x.second;
    }
  }
}

double compute_loss(vector<DataPoint> &pts) {
  // ||Ax-b||
  double loss = 0;
  for (int i = 0; i < pts.size(); i++) {
    map<int, double> row = pts[i].second;
    double est = 0;
    for (auto const &x: row) {
      est += x.second * model[x.first][0];
    }
    loss += (est - B[pts[i].first]) * (est - B[pts[i].first]);
  }
  return loss / (double)N_DIMENSION;
}


void do_cyclades_gradient_descent_with_points(DataPoint *access_pattern, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, int thread_id, vector<int> &batch_pattern) {

  pin_to_core(thread_id);
  int did_last_batch = 0;
  for (int batch_iter = 0; batch_iter < access_length.size(); batch_iter++) {
    int batch = batch_pattern[batch_iter];
    
    //Wait for all threads to be on the same batch
    if (SHOULD_SYNC) {
      thread_batch_on[thread_id] = batch_iter;
      int waiting_for_other_threads = 1;
      while (waiting_for_other_threads) {
	waiting_for_other_threads = 0;
	for (int ii = 0; ii < NTHREAD; ii++) {
	  if (thread_batch_on[ii] < batch_iter) {
	    waiting_for_other_threads = 1;
	    break;
	  }
	}
      }
    }

    //For every data point in the connected component
    for (int i = 0; i < access_length[batch]; i++) {
	int indx = batch_index_start[batch]+i;
	DataPoint p = access_pattern[indx];
	int row = p.first;
	map<int, double> sparse_array = p.second;
	int update_order = order[indx];
	//int update_order;
	//if (did_last_batch) update_order = BATCH_SIZE * (batch_iter-1) + remainder_batch_size + i + 1;
	//else update_order = BATCH_SIZE * batch_iter + i + 1;
	
	if (!SAGA) {

	  //compute gradient
	  double gradient = 0;
	  for (auto const &x : sparse_array) {
	    gradient += x.second * model[x.first][0];
	  }
	  gradient = 2 * (gradient - B[row]);
	  //apply gradient
	  for (auto const & x : sparse_array) {
	    double full_gradient = gradient * x.second;
	    model[x.first][0] -= GAMMA * full_gradient;
	  }
	}
	else {

	  //Update coords
	  for (auto const &x : sparse_array) {
	    int diff = update_order - bookkeeping[x.first] - 1;
	    if (diff < 0 && CRIMP) {
	      diff = 0;
	    }
	    model[x.first][0] -= GAMMA * diff * sum_gradients[x.first] / (double)N_DIMENSION;
	  }

	  //compute gradient
	  double gradient = 0;
	  for (auto const &x : sparse_array) {
	    gradient += x.second * model[x.first][0];
	  }
	  gradient = 2 * (gradient - B[row]);

	  //apply gradient
	  for (auto const & x : sparse_array) {
	    double prev_gradient = prev_gradients[row][x.first];
	    double sum_gradient = sum_gradients[x.first];
	    double full_gradient = (gradient*x.second) - prev_gradient + sum_gradient/(double)N_DIMENSION;
	    model[x.first][0] -= GAMMA * (full_gradient);
	    sum_gradients[x.first] += (gradient*x.second) - prev_gradient;
	    prev_gradients[row][x.first] = gradient*x.second;
	    bookkeeping[x.first] = update_order;
	  }
	}
    }
    if (batch == N_DIMENSION/BATCH_SIZE) did_last_batch = 1;
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
  ///cout << endl;
  
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

void compute_CC_thread(map<int, vector<int> > &CCs, vector<DataPoint> &points, int start, int end, int thread_id, int *tree) {
  pin_to_core(thread_id);
  //int tree[end-start + N_DIMENSION];

  for (long long int i = 0; i < end-start + N_DIMENSION; i++)
    tree[i] = i;

  for (int i = start; i < end; i++) {
    DataPoint p = points[i];
    map<int, double> row = p.second;
    int src = i-start;
    int src_group = union_find(src, tree);
    for (map<int, double>::iterator it = row.begin(); it != row.end(); it++) {
      int element = union_find(it->first + end-start, tree);
      tree[element] = src_group;
    }
  }
  for (int i = 0; i < end-start; i++) {
    int group = union_find(i, tree);
    CCs[group].push_back(i+start);
  }
}

void initialize_model(vector<DataPoint> &sparse_matrix) {

  double randn[N_DIMENSION], model_init[N_DIMENSION];
  for (int i = 0; i < N_DIMENSION; i++) randn[i] = rand() % RANGE;
  mat_vect_mult(sparse_matrix, randn, model_init);

    //Initialize model
  for (int j = 0; j < N_DIMENSION; j++) {
    //model[j][0] = rand() % RANGE;;
    model[j][0] = model_init[j];
    bookkeeping[j] = 0;
    sum_gradients[j] = 0;
  }
}

void update_coords() {
    for (int i = 0; i < N_DIMENSION; i++) {
	int diff = N_DIMENSION - bookkeeping[i];
	if (diff < 0 && CRIMP) diff = 0;
	double sum = 0;
	model[i][0] -= GAMMA * diff * sum_gradients[i] / (double)N_DIMENSION;
    }
}

void clear_bookkeeping() {
  if (SAGA) {
    update_coords();
    for (int i = 0; i < N_DIMENSION; i++) {
      bookkeeping[i] = 0;
    }
  }
}

vector<DataPoint> get_transpose(vector<DataPoint> &mat) {
  vector<DataPoint> t(N_DIMENSION);
  for (int i = 0; i < N_DIMENSION; i++) t[i].first = i;
  for (int i = 0; i < mat.size(); i++) {
    map<int, double> row = mat[i].second;
    for (auto const &x : row) {
      t[x.first].second[mat[i].first] = x.second;
    }
  }
  return t;
}

void initialize_matrix_data(vector<DataPoint> &sparse_matrix) {
#ifdef VERTEX_WEIGHT_FILE
  ifstream in(VERTEX_WEIGHT_FILE);
  for (int i = 0; i < N_DIMENSION; i++) in >> B[i];
#else  
  double randn[N_DIMENSION];
  for (int i = 0; i < N_DIMENSION; i++) randn[i] = 0;rand() % RANGE;
  mat_vect_mult(sparse_matrix, randn, B);
  //add noise
  for (int i = 0; i < N_DIMENSION; i++) {
    B[i] += rand() % RANGE;
  }
#endif
}

vector<DataPoint> get_sparse_matrix() {

  vector<DataPoint> A(N_DIMENSION);
  for (int i = 0; i < N_DIMENSION; i++) A[i].first = i;
  ifstream in(MATRIX_DATA_FILE);
  string s;
  map<int, int> node_id_map;
  while (getline(in, s)) {
    stringstream linestream(s);
    if (s[0] == '#' || s[0] == '%') continue;
    int n1, n2;
    double w = 1;
    linestream >> n1 >> n2;
    if (node_id_map.find(n1) == node_id_map.end())
      node_id_map[n1] = node_id_map.size();
    if (node_id_map.find(n2) == node_id_map.end())
      node_id_map[n2] = node_id_map.size();
    if (linestream >> w);
    else w = 1;
    A[node_id_map[n1]].second[node_id_map[n2]] = w;
    A[node_id_map[n2]].second[node_id_map[n1]] = w;
  }

  initialize_matrix_data(A);
  return A;
}

vector<DataPoint> get_sparse_matrix_synthetic() {
  vector<DataPoint> sparse_matrix;

  //Randomize the sparse matrix
  for (int i = 0; i < N_DIMENSION; i++) {
    DataPoint p = DataPoint(i, map<int, double>());
    /*for (int j = 0; j < rand() % NUM_SPARSE_ELEMENTS_IN_ROW + 1; j++) {
      int column = rand() % N_DIMENSION;
      if (p.second.find(column) == p.second.end()) {
	p.second[column] = rand() % RANGE;
	}*/
    
    //} 
    p.second[i] = rand() % RANGE;
    sparse_matrix.push_back(p);
  }
  
  initialize_matrix_data(sparse_matrix);

  return sparse_matrix;
}

void cyc_least_squares() {
  vector<DataPoint> points = get_sparse_matrix();
  initialize_model(points);
  random_shuffle(points.begin(), points.end());
  vector<DataPoint> points_copy(points);
  
  Timer overall;

  int n_batches = (int)ceil((points.size() / (double)BATCH_SIZE));
  vector<vector<DataPoint> > access_pattern(NTHREAD);
  vector<vector<int > > access_length(NTHREAD);
  vector<vector<int > > batch_index_start(NTHREAD);
  vector<vector<int > > order(NTHREAD);

  for (int i = 0; i < NTHREAD; i++) {
    access_length[i].resize(n_batches);
    batch_index_start[i].resize(n_batches);
    order[i].resize(n_batches);
  }

  vector<int> batch_pattern;
  for (int i = 0; i < n_batches; i++) batch_pattern.push_back(i);

  //CC Parallel
  map<int, vector<int> > CCs[n_batches];
#pragma omp parallel for
  for (int i = 0; i < n_batches; i++) {
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());
    if (i == n_batches-1) remainder_batch_size = end-start;
    compute_CC_thread(CCs[i], points, start, end, omp_get_thread_num(), trees[omp_get_thread_num()]);
  }

  for (int i = 0; i < n_batches; i++) {
    distribute_ccs(CCs[i], access_pattern, access_length, batch_index_start, i, points, order);
  }

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
      do_cyclades_gradient_descent_with_points((DataPoint *)&access_pattern[0][0], access_length[0], batch_index_start[0], order[0], 0, batch_pattern);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, (DataPoint *)&access_pattern[j][0], ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j, ref(batch_pattern)));
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
    
    //random_shuffle(batch_pattern.begin(), batch_pattern.end());
  }
  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
      cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
}

void hog_least_squares() {
  vector<DataPoint> points = get_sparse_matrix();
  initialize_model(points);
  random_shuffle(points.begin(), points.end());
  Timer overall;

  //Hogwild access pattern construction
  vector<vector<DataPoint> > access_pattern(NTHREAD);
  vector<vector<int > > access_length(NTHREAD);
  vector<vector<int> > batch_index_start(NTHREAD);
  vector<vector<int> > order(NTHREAD);
  vector<int> batch_pattern;
  batch_pattern.push_back(0);

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
      do_cyclades_gradient_descent_with_points((DataPoint *)&access_pattern[0][0], access_length[0], batch_index_start[0], order[0], 0, batch_pattern);
    }
    else {
      for (int j = 0; j < NTHREAD; j++) {
	threads.push_back(thread(do_cyclades_gradient_descent_with_points, (DataPoint *)&access_pattern[j][0], ref(access_length[j]), ref(batch_index_start[j]), ref(order[j]), j, ref(batch_pattern)));
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
  std::cout << std::setprecision(10);
  pin_to_core(0);

  /*sum_gradients = (double **)malloc(sizeof(double *) * N_DIMENSION);
  //model = (double **)malloc(sizeof(double *) * N_DIMENSION);
  for (int i = 0; i < N_DIMENSION; i++) {
    //model[i] = (double *)malloc(sizeof(double) * N_CATEGORIES_CACHE_ALIGNED);
    sum_gradients[i] = (double *)malloc(sizeof(double) * N_CATEGORIES_CACHE_ALIGNED);
  }

  for (int i = 0; i < N_DIMENSION; i++) {
    bookkeeping[i] = 0;
    for (int j = 0; j < N_CATEGORIES_CACHE_ALIGNED; j++) {
      sum_gradients[i][j] = 0;
    }
    }*/

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

  //Allocate memory for union find
  for (int i = 0; i < NTHREAD; i++) {
    trees[i] = (int *)malloc(sizeof(int) * (BATCH_SIZE + N_DIMENSION));
  }

  //Clear miscellaneous datastructures for CC load balancing
  for (int i = 0; i < NTHREAD; i++) {
    cur_bytes_allocated[i] = 0;
    cur_datapoints_used[i] = 0;
    thread_batch_on[i] = 0;
  }

  if (HOG) {
      hog_least_squares();
  }
  if (CYC) {
    cyc_least_squares();
  }
  //for (int i = 0; i < N_DIMENSION; i++) {
    //cout << model[i][0] << " ";
  //}
  //cout << endl;
  //for (int i = 0; i < NTHREAD; i++)
  //cout << thread_load_balance[i] << endl;
  //for (int i = 0; i < N_DIMENSION; i++) cout << model[i][0] << endl;
}
