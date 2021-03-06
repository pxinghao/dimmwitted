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
#define K 30

#define N_EPOCHS 200
#define WORD_EMBEDDINGS_FILE "sparse_graph"
//#define N_NODES 111
//#define WORD_EMBEDDINGS_FILE "mat_comp_dat"
#define N_NODES 105
#define N_DATAPOINTS 1096
#ifndef SHUFFLE
#define SHUFFLE 0
#endif

#ifndef SAGA
#define SAGA 1
#endif

double sum_grad[N_NODES][K], prev_grad[N_DATAPOINTS][N_NODES][K];
double model[N_NODES][K] __attribute__((aligned(64)));
int terminal_nodes[K];
double GAMMA = 2e-3; //SAGA
//double GAMMA = 4e-4; //SAGA
//double GAMMA = 3e-3; //SGD
double C = 0;

using namespace std;

typedef tuple<int, int, double> DataPoint;

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
  //return loss;
  return loss / points.size();
}

int cmp(const void *a, const void *b) {
  double first = *(double *)a;
  double second = *(double *)b;
  if (first > second) return -1;
  return 1;
}

void project_constraint(double *vec) {

  double sorted[K];
  memcpy(sorted, vec, sizeof(double)*K);
  qsort(sorted, K, sizeof(double), cmp);
  
  double sum = 0, chosen_sum = 0;
  int p = 0;
  for (int i = 0; i < K; i++) {
    sum += sorted[i];
    if (sorted[i] - (1 / (double)(i+1)) * (sum - 1) > 0) {
      p = i+1;
      chosen_sum = sum;
    }
  }
  double theta = (1 / (double)p) * (chosen_sum - 1);
  for (int i = 0; i < K; i++) {
    vec[i] = max((double)0, (double)vec[i]-(double)theta);
  }
}

void initialize_model() {
  for (int i = 0; i < N_DATAPOINTS; i++) {
    for (int j = 0; j < N_NODES; j++) {
      for (int k = 0; k < K; k++) {
	prev_grad[i][j][k] = 0;
      }
    }
  }
  for (int i = 0; i < N_NODES; i++) {
    for (int j = 0; j < K; j++) {
      model[i][j] = rand() / (double)RAND_MAX;
      //model[i][j] = .5;
    }
  }
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

vector<DataPoint> get_mat_compl_data() {
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

int main(void) {
  srand(0);
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(20);

  vector<DataPoint> pts = get_word_embeddings_data();
  initialize_model();
  //random_shuffle(pts.begin(), pts.end());

  for (int i = 0; i < N_NODES; i++) {
    for (int j = 0; j < K; j++) {
      sum_grad[i][j] = 0;
    }
  }

  for (int i = 0; i < N_DATAPOINTS; i++) {
    for (int j = 0; j < N_NODES; j++) {
      for (int k = 0; k < K; k++) {
	prev_grad[i][j][k] = 0;
      }
    }
  }

  int n_datapoints_seen = 0;

  vector<int> pattern(pts.size());
  for (int i = 0; i < pts.size(); i++) pattern[i] = i;

  //Epoch loop
  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    cout << compute_loss(pts) << " 0 0" << endl;

    if (SHUFFLE)
      random_shuffle(pattern.begin(), pattern.end());
    
    //For each sampled datapoint
    for (int  ii = 0; ii < pts.size(); ii++) {

      //Compute the gradient at every coordinate in the model
      int pt_index = pattern[ii];
      DataPoint pt = pts[pt_index];
      //DataPoint pt = pts[rand() % pts.size()];
      int first_coord = get<0>(pt);
      int second_coord = get<1>(pt);
      int weight = get<2>(pt);
      n_datapoints_seen++;

      //Compute the gradient at the point
      double first_grad[K], second_grad[K];
      double l2norm_sqr = 0;
      for (int j = 0; j < K; j++) {
	l2norm_sqr += (model[first_coord][j] + model[second_coord][j]) * (model[first_coord][j] + model[second_coord][j]);
      }
      double mult = 2 * weight * (log(weight) - l2norm_sqr - C);

      for (int j = 0; j < K; j++) {
	second_grad[j] = first_grad[j] = -1 * (mult * 2 * (model[first_coord][j] + model[second_coord][j]));
	//first_grad[j] = 2 * weight * (model[first_coord][j] - model[second_coord][j]);
	//second_grad[j] = -first_grad[j];
      }

      //Create the gradient matrix
      double gradient[N_NODES][K];
      for (int coord = 0; coord < N_NODES; coord++) {
	if (coord == first_coord) {
	  for (int i = 0; i < K; i++) gradient[coord][i] = first_grad[i];
	}
	else if (coord == second_coord) {
	  for (int i = 0; i < K; i++) gradient[coord][i] = second_grad[i];
	}
	else {
	  for (int i = 0; i < K; i++) gradient[coord][i] = 0;
	}
      }

      //Compute the full gradient update matrix for SAG
      double full_gradient[N_NODES][K];
      for (int i = 0; i < N_NODES; i++) {
	for (int j = 0; j < K; j++) {
	  
	  if (SAGA) {
	    if (epoch != 0) {
	      //Full gradient = (cur_grad - prev_grad + sum_grad) / n
	      full_gradient[i][j] = (gradient[i][j] - prev_grad[pt_index][i][j] + sum_grad[i][j] / N_DATAPOINTS);
	      //Update model
	      model[i][j] -= GAMMA * full_gradient[i][j];
	      //Update sum
	      sum_grad[i][j] += -prev_grad[pt_index][i][j] + gradient[i][j];
	      //Update previous gradient
	      prev_grad[pt_index][i][j] = gradient[i][j];
	    }
	    else {
	      model[i][j] -= GAMMA * gradient[i][j];
	      sum_grad[i][j] += -prev_grad[pt_index][i][j] + gradient[i][j];
	      prev_grad[pt_index][i][j] = gradient[i][j];
	    }
	  }
	  else {
	    model[i][j] -= GAMMA * gradient[i][j];
	  }
	}
	//Project constraints
	//project_constraint((double *)model[i]);
      }

	for (int j = 0; j < K; j++) {
	  //cout << model[first_coord][j] << endl;
	  //cout << model[second_coord][j] << endl;
	  //cout << sum_grad[first_coord][j] << endl;
	  //cout << prev_grad[pt_index][first_coord][j] << endl;
	}
    }
  }
}
