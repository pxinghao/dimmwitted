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
//#include <numa.h>
#include <sched.h>
#include <iomanip>
#include <mutex>
#include <omp.h>
#include <cmath>

#define RANGE 10
#define N_EPOCHS 10
#define N_DIMENSION 200
#define NUM_SPARSE_ELEMENTS_IN_ROW 2

using namespace std;

//row #, sparse_vector of row
typedef pair<int, map<int, double> > DataPoint;

double model[N_DIMENSION];
double B[N_DIMENSION];
double norm_row[N_DIMENSION];
double C = 0; //1 / (frobenius norm of matrix)^2
double LAMBDA = 0;
//double GAMMA = 2e-15;
double GAMMA = 1;

double gradient_tilde[N_DIMENSION][N_DIMENSION];
double sum_gradient_tilde[N_DIMENSION];

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
      ai_t_x += model[element.first]* element.second;      
    }
    first -= ai_t_x * ai_t_x;
    loss += .5 * first - 1/(double)N_DIMENSION * second;
  }
  return loss;
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

void print_22(vector<DataPoint> &mat) {
  cout << mat[0].second[0] << " " <<mat[0].second[1] <<endl;
    cout << mat[1].second[0] << " " <<mat[1].second[1] <<endl;
}

vector<DataPoint> get_transpose(vector<DataPoint> &mat) {
  vector<DataPoint> t(N_DIMENSION);
  for (int i = 0; i < N_DIMENSION; i++) t[i].first = i;
  for (int i = 0; i < mat.size(); i++) {
    map<int, double> row = mat[i].second;
    for (auto const &x : row) {
      t[x.first].second[i] = x.second;
    }
  }
  return t;
}

void initialize_matrix_data(vector<DataPoint> &sparse_matrix) {
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
    C = 1/sum;

    //Initialize B to be random
    double random_d[N_DIMENSION];
    double extra_b[N_DIMENSION];
    double sum_b = 0;
    for (int i = 0; i < N_DIMENSION; i++) {
      random_d[i] = rand() % RANGE;
    }
    //compute AA*random_vector
    memset(B, 0, sizeof(double)*N_DIMENSION);

    mat_vect_mult(sparse_matrix, random_d, B);
    mat_vect_mult(sparse_matrix, B, extra_b);
    for (int i = 0; i < N_DIMENSION; i++) {
      sum_b += extra_b[i]*extra_b[i];
    }
    double norm_b = sqrt(sum_b);
    for (int i = 0; i < N_DIMENSION; i++) {
      B[i] = extra_b[i]/norm_b;
    }
    
    //Compute Lambda
    vector<DataPoint> transpose = get_transpose(sparse_matrix);
    double x_k_prime[N_DIMENSION], x_k[N_DIMENSION], x_k_prime_prime[N_DIMENSION];
    memset(x_k_prime, 0, sizeof(double) * N_DIMENSION);
    memcpy(x_k, random_d, sizeof(double) * N_DIMENSION);
    for (int i = 0; i < 3; i++) {
      mat_vect_mult(sparse_matrix, x_k, x_k_prime);
      mat_vect_mult(transpose, x_k_prime, x_k);

      double x_3_norm = 0;
      for (int i = 0; i < N_DIMENSION; i++) {
	x_3_norm += x_k[i] * x_k[i];
      }
      for (int i = 0; i < N_DIMENSION; i++) {
	x_k[i] /= sqrt(x_3_norm);
      }
      memset(x_k_prime, 0, sizeof(double) * N_DIMENSION);      
    }
    mat_vect_mult(sparse_matrix, x_k, x_k_prime);
    mat_vect_mult(transpose, x_k_prime, x_k_prime_prime);
    for (int i = 0; i < N_DIMENSION; i++) {
      LAMBDA += x_k_prime_prime[i] * 1.1 * x_k[i];
    }
}

vector<DataPoint> get_sparse_matrix_synthetic() {
  vector<DataPoint> sparse_matrix;

  //Initialize # of zeroes in columns
  for (int i = 0; i < N_DIMENSION; i++) {
    //num_zeroes_in_column[i] = N_DIMENSION;
  }
  
  //Randomize the sparse matrix
  for (int i = 0; i < N_DIMENSION; i++) {
    DataPoint p = DataPoint(i, map<int, double>());
    for (int j = 0; j < rand() % NUM_SPARSE_ELEMENTS_IN_ROW + 1; j++) {
      int column = rand() % N_DIMENSION;
      if (p.second.find(column) == p.second.end()) {
	p.second[column] = rand() % RANGE;
	//num_zeroes_in_column[column]--;
      }
    } 
    sparse_matrix.push_back(p);
  }
  
  initialize_matrix_data(sparse_matrix);

  return sparse_matrix;
}

void initialize_model() {

    //Initialize model
    for (int j = 0; j < N_DIMENSION; j++) {
      model[j] = B[j];
	sum_gradient_tilde[j] = 0;
    }
}

void calculate_gradient_tilde(vector<DataPoint> &pts) {

    for (int i = 0; i < N_DIMENSION; i++) {
      get_gradient(pts[i], gradient_tilde[pts[i].first]);
    }

    for (int i = 0; i < N_DIMENSION; i++) {
	sum_gradient_tilde[i] = 0;
	for (int j = 0; j < N_DIMENSION; j++) {
	    sum_gradient_tilde[i] += gradient_tilde[j][i];
	}
    }
}

int main(void) {
    srand(0);
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(10);
    vector<DataPoint> sparse_matrix = get_sparse_matrix_synthetic();
    initialize_model();
    random_shuffle(sparse_matrix.begin(), sparse_matrix.end());
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
	calculate_gradient_tilde(sparse_matrix);

	cout << compute_loss(sparse_matrix) << endl;
	vector<int> random_pattern;
	for (int i = 0; i < sparse_matrix.size(); i++) 
	  random_pattern.push_back(i);
	//random_shuffle(random_pattern.begin(), random_pattern.end());
	//for (int row = 0; row < sparse_matrix.size(); row++) {
	for (int access = 0; access < random_pattern.size(); access++) {
	    int row = random_pattern[access];
	    DataPoint sparse_row = sparse_matrix[row];
	    double gradient[N_DIMENSION];
	    get_gradient(sparse_row, gradient);
	    for (int i = 0; i < N_DIMENSION; i++) {
	      if (sparse_row.second.find(i) != sparse_row.second.end()) {
		//cout << "MOD: " << model[i] << endl;
	      }
	      double full_gradient = gradient[i] - gradient_tilde[sparse_row.first][i] + sum_gradient_tilde[i] / N_DIMENSION;
	      model[i] -= GAMMA * full_gradient;
	      if (sparse_row.second.find(i) != sparse_row.second.end()) {
		//cout << "B : " << B[i] << endl;
		//cout << "mod GRAD: " << gradient_tilde[sparse_row.first][i] << endl;
		//cout << "MOD: " << model[i] << endl;
	      }
	    }
	    //exit(0);
	}
	//for (int j = 0; j < N_DIMENSION; j++) cout << model[j] << endl;
    }
}
