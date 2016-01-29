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

#define N_DIMENSION 10000
#define N_DIMENSION_CACHE_ALIGNED (N_DIMENSION/8+1) * 8
#define NUM_SPARSE_ELEMENTS_IN_ROW 10

#ifndef NTHREAD
#define NTHREAD 8
#endif

#define RANGE 100000

#ifndef N_EPOCHS
#define N_EPOCHS 100
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 1000
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
#define START_GAMMA 2e-9
#endif

#ifndef SVRG
#define SVRG 1
#endif

double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

int volatile thread_batch_on[NTHREAD];

//Problem: minimize datapoints * model - B

double model[N_DIMENSION];
double B[N_DIMENSION];
double C = 0; //1 / (frobenius norm of matrix)^2
double LAMBDA = 0;

//precompute sum of powers of lambda * C
double sum_pows[N_DIMENSION];

//double gradient_tilde[N_DIMENSION][N_DIMENSION];
double **gradient_tilde;
//double gradient_tilde
double sum_gradient_tilde[N_DIMENSION];
double model_tilde[N_DIMENSION];

int bookkeeping[N_DIMENSION];

double thread_load_balance[NTHREAD];
size_t cur_bytes_allocated[NTHREAD];
int cur_datapoints_used[NTHREAD];
int core_to_node[NTHREAD];

using namespace std;

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

void get_gradient(DataPoint &p, double *out) {
    double ai_t_x = 0;
    for (int i = 0; i < N_DIMENSION; i++) {
	double weight = 0;
	if (p.second.find(i) != p.second.end()) weight = p.second[i];
	ai_t_x += weight * model[i];
	out[i] = LAMBDA * C * model[i];
    }
    for (int i = 0; i < N_DIMENSION; i++) {
	double weight = 0;
	if (p.second.find(i) != p.second.end()) weight = p.second[i];
	out[i] -= ai_t_x * weight + B[i] / N_DIMENSION;
    }
}

double compute_loss(vector<DataPoint> &pts) {
  //Compute gradient loss
  /*double loss = 0;
  for (int i = 0; i < N_DIMENSION; i++) {
    double grad[N_DIMENSION];
    get_gradient(pts[i], grad);
    for (int i = 0; i < N_DIMENSION; i++)
      loss += grad[i] * grad[i];
  }
  return loss / N_DIMENSION;*/
  double loss = 0;
  double second = 0;
  double sum_sqr = 0;
  for (int j = 0; j < N_DIMENSION; j++) {
    second += model[j] * B[j];
    sum_sqr += model[j] * model[j];
  }
  for (int i = 0; i < pts.size(); i++) {
    double ai_t_x = 0;
    map<int, double> sparse_row = pts[i].second;
    double first = sum_sqr * C * LAMBDA;
    for (auto const & element : sparse_row) {
      ai_t_x += model[element.first] * element.second;      
    }
    first -= ai_t_x * ai_t_x;
    loss += .5 * first - 1/(double)N_DIMENSION * second;
  }
  return loss;
}

void calculate_gradient_tilde(vector<DataPoint> &pts) {

    memcpy(model_tilde, model, sizeof(double) * N_DIMENSION);
    for (int i = 0; i < N_DIMENSION; i++) {
      get_gradient(pts[i], gradient_tilde[i]);
    }

    for (int i = 0; i < N_DIMENSION; i++) {
	sum_gradient_tilde[i] = 0;
	for (int j = 0; j < N_DIMENSION; j++) {
	    sum_gradient_tilde[i] += gradient_tilde[j][i];
	}
    }
}

void do_cyclades_gradient_descent_with_points(DataPoint *access_pattern, vector<int> &access_length, vector<int> &batch_index_start, vector<int> &order, int thread_id, vector<int> &batch_pattern) {
  pin_to_core(thread_id);
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
	
	if (!SVRG) {
	  double out[N_DIMENSION];
	  double ai_t_x = 0;
	  for (int i = 0; i < N_DIMENSION; i++) {
	    double weight = 0;
	    if (p.second.find(i) != p.second.end()) weight = p.second[i];
	    ai_t_x += weight * model[i];
	    out[i] = LAMBDA * C * model[i];
	  }
	  for (int i = 0; i < N_DIMENSION; i++) {
		double weight = 0;
		if (p.second.find(i) != p.second.end()) weight = p.second[i];
		out[i] -= ai_t_x * weight + B[i] / N_DIMENSION;
	  }
	  for (int i = 0; i < N_DIMENSION; i++) {
	    model[i] -= GAMMA * out[i];
	  }
	}
	else {
	  //catch up
	  for (auto const &x : sparse_array) {
	    int diff = update_order - bookkeeping[x.first] - 1;
	    diff = max(diff, 0);
	    double sum = sum_pows[(int)diff];
	    double first_part = model[x.first] * pow(1 - LAMBDA * C * GAMMA, diff);
	    double second_part = GAMMA * (LAMBDA*C*model_tilde[x.first] - 1/(double)N_DIMENSION*sum_gradient_tilde[x.first]) * sum;
	    model[x.first] = first_part + second_part;
	  }
	  
	  //Compute gradient
	  double ai_t_x = 0;
	  for (auto const &x : sparse_array) {
	    ai_t_x += model[x.first] * x.second;
	  }
	  double ai_ai_t_x = 0;
	  for (auto const &x : sparse_array) {
	    double gradient = C * LAMBDA * model[x.first] - ai_t_x * x.second - B[x.first] / N_DIMENSION;
	    model[x.first] -= GAMMA * (gradient - gradient_tilde[row][x.first] + sum_gradient_tilde[x.first] / N_DIMENSION);
	    bookkeeping[x.first] = update_order;
	  }
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
    cout << balances[i].second << " ";
  }
  cout << endl;

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
  int tree[end-start + N_DIMENSION];

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

void initialize_model() {

    //Initialize model
    for (int j = 0; j < N_DIMENSION; j++) {
	model[j] = rand() % 100;
	bookkeeping[j] = 0;
	sum_gradient_tilde[j] = 0;
    }
}

void update_coords() {
    for (int i = 0; i < N_DIMENSION; i++) {
	double diff = N_DIMENSION - bookkeeping[i];
	double sum = 0;
	for (int j = 0; j < diff; j++) {
	    sum += pow(1 - GAMMA * LAMBDA * C, j);
	}
	double first_part = model[i] * pow(1 - LAMBDA * C * GAMMA, diff);
	double second_part = GAMMA * (LAMBDA*C*model_tilde[i] - 1/(double)N_DIMENSION*sum_gradient_tilde[i]) * sum;
	model[i] = first_part + second_part;
    }
}

void clear_bookkeeping() {
    if (SVRG) {
	update_coords();
	for (int i = 0; i < N_DIMENSION; i++) {
	    bookkeeping[i] = 0;
	}
    }
}

void mat_vect_mult(vector<DataPoint> &sparse_matrix, double *in, double *out) {
    for (int j = 0; j < sparse_matrix.size(); j++) {
	map<int, double> sparse_row = sparse_matrix[j].second;
	int row = sparse_matrix[j].first;
	for (auto const & x : sparse_row) {
	    out[row] += in[x.first] * x.second;
	}
    }
}

vector<DataPoint> get_sparse_matrix() {
    vector<DataPoint> sparse_matrix;

    //Randomize the sparse matrix
    for (int i = 0; i < N_DIMENSION; i++) {
      DataPoint p = DataPoint(i, map<int, double>());
      for (int j = 0; j < rand() % NUM_SPARSE_ELEMENTS_IN_ROW; j++) {
	p.second[rand() % N_DIMENSION] = rand() % RANGE;
      } 
      sparse_matrix.push_back(p);
    }

    //Normalize the rows to be 1
    for (int i = 0; i < N_DIMENSION; i++) {
	map<int, double> &row = sparse_matrix[i].second;
	double sum = 0;
	for (auto const & x : row) {
	    sum += x.second * x.second;
	}
	double norm_factor = sqrt(sum);
	if (norm_factor != 0) {
	  for (auto const & x : row) {
	    row[x.first] /= norm_factor;
	  }
	}
    }

    //Calculate frobenius norm of matrix
    double sum = 0;
    for (int i = 0; i < N_DIMENSION; i++) {
      map<int, double> row = sparse_matrix[i].second;
      for (auto const & x : row) {
	sum += x.second * x.second;
      }
    }
    C = 1 / sum;

    //Initialize B to be random
    for (int i = 0; i < N_DIMENSION; i++) {
      B[i] = rand() % 100;
    }
    
    //Compute Lambda
    double x_k_prime[N_DIMENSION], x_k[N_DIMENSION];
    memset(x_k_prime, 0, sizeof(double) * N_DIMENSION);
    memcpy(x_k, B, sizeof(double) * N_DIMENSION);
    for (int i = 0; i < 3; i++) {
      mat_vect_mult(sparse_matrix, x_k, x_k_prime);
      memcpy(x_k, x_k_prime, sizeof(double) * N_DIMENSION);
      memset(x_k_prime, 0, sizeof(double) * N_DIMENSION);
    }
    double x_3_norm = 0;
    for (int i = 0; i < N_DIMENSION; i++) {
      x_3_norm += x_k[i] * x_k[i];
    }
    x_3_norm = sqrt(x_3_norm);
    mat_vect_mult(sparse_matrix, x_k, x_k_prime);
    for (int i = 0; i < N_DIMENSION; i++) {
      LAMBDA += x_k_prime[i] * 1.1 * x_3_norm;
    }

    //precompute sum of powers
    sum = 0;
    for (int j = 0; j < N_DIMENSION; j++) {
      sum += pow(1 - LAMBDA * GAMMA * C,  j);
      sum_pows[j] = sum;
    }
    
    return sparse_matrix;
}

void cyc_matrix_inverse() {
  vector<DataPoint> points = get_sparse_matrix();
  initialize_model();
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
  //#pragma omp parallel for
  for (int i = 0; i < n_batches; i++) {
    int start = i * BATCH_SIZE;
    int end = min((i+1)*BATCH_SIZE, (int)points.size());
    compute_CC_thread(CCs[i], points, start, end, omp_get_thread_num());
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

    calculate_gradient_tilde(points);
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
    
    //Shuffle
    for (int ii = 0; ii < NTHREAD; ii++) {
      random_shuffle(order[ii].begin(), order[ii].end());
    }
  }
  if (!SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) {
    cout << overall.elapsed() << endl;
    cout << gradient_time.elapsed() << endl;
    cout << compute_loss(points) << endl;
  }
}

void hog_matrix_inverse() {
  vector<DataPoint> points = get_sparse_matrix();
  initialize_model();
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
    
    calculate_gradient_tilde(points);

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

    
    for (int j = 0; j < NTHREAD; j++) {
      random_shuffle(access_pattern[j].begin(), access_pattern[j].end());
    }
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

  gradient_tilde = (double **)malloc(sizeof(double *) * N_DIMENSION);
  for (int i = 0; i < N_DIMENSION; i++) {
    gradient_tilde[i] = (double *)malloc(sizeof(double) * N_DIMENSION_CACHE_ALIGNED);
  }

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

  //Clear miscellaneous datastructures for CC load balancing
  for (int i = 0; i < NTHREAD; i++) {
    cur_bytes_allocated[i] = 0;
    cur_datapoints_used[i] = 0;
    thread_batch_on[i] = 0;
  }

  if (HOG) {
      hog_matrix_inverse();
  }
  if (CYC) {
    cyc_matrix_inverse();
  }
  //for (int i = 0; i < NTHREAD; i++)
  //cout << thread_load_balance[i] << endl;
}
