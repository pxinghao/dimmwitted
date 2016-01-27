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
#include <omp.h>

#include <iostream>
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
#define NTHREAD 8
#endif

#ifndef N_EPOCHS
#define N_EPOCHS 10
#endif 

#ifndef BATCH_SIZE
#define BATCH_SIZE 400
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
#define START_GAMMA 1e-3// SAGA
//#define START_GAMMA 2 // SGD
#endif

double GAMMA = START_GAMMA;
double GAMMA_REDUCTION = 1;

int volatile thread_batch_on[NTHREAD];

//double ** prev_gradients[NTHREAD];
//double ** model, **sum_gradients;
double prev_gradients[N_DATAPOINT][N_COORDS][N_CATEGORIES];
double model[N_COORDS][N_CATEGORIES];
double sum_gradients[N_COORDS][N_CATEGORIES];
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

int main(void) {


  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(20);

  //Read data and initialize model
  srand(100);
  vector<DataPoint> pts = get_text_classification_data();
  //random_shuffle(pts.begin(), pts.end());
  initialize_model();

  //Epoch loop
  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    cout << compute_loss(pts) << endl;
    
    //For each sampled datapoint
    for (int pt_index = 0; pt_index < pts.size(); pt_index++) {

      //Compute the gradient at every coordinate in the model
      DataPoint pt = pts[pt_index];
      vector<pair<int, double> > sparse_vector = pt.second;
      
      for (int i = 0; i < sparse_vector.size(); i++) {
	for (int j = 0; j < N_CATEGORIES; j++) {
	  //cout << sparse_vector[i].first << endl;
	  //cout << prev_gradients[pt_index][first_coord][j] << endl;
	  //cout << sum_gradients[first_coord][j] << endl;
	  //cout << model[sparse_vector[i].first][j] << endl;
	  //cout << model[second_coord][j] << endl;
	}
      }
     
      //Compute gradients
      double probs[N_CATEGORIES];

      //Create the gradient matrix
      double gradient[N_COORDS][N_CATEGORIES];
      for (int coord = 0; coord < N_COORDS; coord++) {
	for (int category = 0; category < N_CATEGORIES; category++) {
	  gradient[coord][category] = 0;
	}
      }

      compute_probs(pt, probs);
      
      for (int i = 0; i < sparse_vector.size(); i++) {	
	for (int j = 0; j < N_CATEGORIES; j++) {
	  int is_category = j == pt.first ? 1 : 0;
	  gradient[sparse_vector[i].first][j] = -sparse_vector[i].second * (is_category - probs[j]);
	}
      }      

      //Compute the full gradient update matrix for SAG
      double full_gradient[N_COORDS][N_CATEGORIES];
      for (int i = 0; i < N_COORDS; i++) {
	for (int j = 0; j < N_CATEGORIES; j++) {
	  //model[i][j] -= GAMMA * gradient[i][j];
	  //Full gradient = (cur_grad - prev_grad + sum_gradients) / n
	  full_gradient[i][j] = (gradient[i][j] - prev_gradients[pt_index][i][j] + sum_gradients[i][j] / N_DATAPOINT);
	  //Update model
	  model[i][j] -= GAMMA * full_gradient[i][j];
	  //Update sum
	  sum_gradients[i][j] += -prev_gradients[pt_index][i][j] + gradient[i][j];
	  //Update previous gradient
	  prev_gradients[pt_index][i][j] = gradient[i][j];
	}
      }
    }
  }
}
