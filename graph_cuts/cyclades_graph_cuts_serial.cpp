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
#define K 2

#define N_EPOCHS 20
#define GRAPH_CUTS_FILE "test_case"
//#define N_NODES 110594 + 1 //tsukuba dataset
//#define N_DATAPOINTS 514483 //tsukuba dataset
#define N_NODES 10
#define N_DATAPOINTS 10

double sum_grad[N_NODES][K], prev_grad[N_DATAPOINTS][N_NODES][K];
double model[N_NODES][K] __attribute__((aligned(64)));
int terminal_nodes[K];
double GAMMA = 8e-5;

using namespace std;

typedef tuple<int, int, double> DataPoint;

double compute_loss(vector<DataPoint> points) {
  double loss = 0;
  for (int i = 0; i < points.size(); i++) {
    int u = get<0>(points[i]), v = get<1>(points[i]);
    double w = get<2>(points[i]);
    double sub_loss = 0;
    for (int j = 0; j < K; j++) {
      sub_loss += abs(model[u][j] - model[v][j]);
      //sub_loss += (model[u][j]-model[v][j]) *  (model[u][j]-model[v][j]);
    }
    loss += sub_loss * w;
  }
  return loss / (double)points.size();
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

int is_anchor(int coord) {
  for (int i = 0; i < K; i++) {
    if (terminal_nodes[i] == coord) return 1;
  }
  return 0;
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
      model[i][j] = 0;
    }
  }
  for (int i = 0; i < K; i++) {
    terminal_nodes[i] = i+1;
  }
  for (int i = 0; i < K; i++) {
    model[terminal_nodes[i]-1][i-1] = 1;
  }
}

vector<DataPoint> get_graph_cuts_data() {
  vector<DataPoint> datapoints;
  ifstream in(GRAPH_CUTS_FILE);
  string s;
  int max_node = 0;
  while (getline(in, s)) {
    stringstream linestream(s);
    char command_type;
    linestream >> command_type;
    if (command_type == 'a') {
      int n1, n2;
      double cap;
      linestream >> n1 >> n2 >> cap;
      datapoints.push_back(DataPoint(n1, n2, cap));
      max_node = max(max_node, max(n1, n2));
    }
  }
  return datapoints;
}

int main(void) {

  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(20);

  //Read data and initialize model
  srand(100);
  vector<DataPoint> pts = get_graph_cuts_data();
  //random_shuffle(pts.begin(), pts.end());
  initialize_model();

  //Epoch loop
  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    cout << compute_loss(pts) << endl;
    
    //For each sampled datapoint
    for (int pt_index = 0; pt_index < pts.size(); pt_index++) {

      //Compute the gradient at every coordinate in the model
      DataPoint pt = pts[pt_index];
      int first_coord = get<0>(pt);
      int second_coord = get<1>(pt);
      int weight = get<2>(pt);

      //Check if coordinate is anchor
      int is_first_anchor = is_anchor(first_coord);
      int is_second_anchor = is_anchor(second_coord);

      for (int j = 0; j < K; j++) {
	//cout << prev_grad[pt_index][first_coord][j] << endl;
	//cout << sum_grad[first_coord][j] << endl;
	//cout << model[first_coord][j] << endl;
	//cout << model[second_coord][j] << endl;
      }
      
      //Compute the gradient at the point
      double first_grad[K], second_grad[K];
      for (int i = 0; i < K; i++) {
	first_grad[i] = (model[first_coord][i] - model[second_coord][i] < 0) ? -weight : weight;
	second_grad[i] = -first_grad[i];
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
	if (i == first_coord && is_first_anchor) continue;
	if (i == second_coord && is_second_anchor) continue;
	for (int j = 0; j < K; j++) {
	  //model[i][j] -= GAMMA * gradient[i][j];
	  //Full gradient = (cur_grad - prev_grad + sum_grad) / n
	  full_gradient[i][j] = (gradient[i][j] - prev_grad[pt_index][i][j] + sum_grad[i][j]) / N_DATAPOINTS;
	  //Update model
	  model[i][j] -= GAMMA * full_gradient[i][j];
	  //Update sum
	  sum_grad[i][j] += -prev_grad[pt_index][i][j] + gradient[i][j];
	  //Update previous gradient
	  prev_grad[pt_index][i][j] = gradient[i][j];
	}
	//Project constraints
	project_constraint((double *)model[i]);
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
