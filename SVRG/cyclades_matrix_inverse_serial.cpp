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

#define N_EPOCHS 20
#define N_DIMENSION 1000
#define NUM_SPARSE_ELEMENTS_IN_ROW 20

using namespace std;

//row #, sparse_vector of row
typedef pair<int, map<int, double> > DataPoint;

double model[N_DIMENSION];
double B[N_DIMENSION];
double norm_row[N_DIMENSION];
double C = 0; //1 / (frobenius norm of matrix)^2
double LAMBDA = 0;
//double GAMMA = 2e-15;
double GAMMA = 2e-13;

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
    for (int i = 0; i < N_DIMENSION; i++) {
	double grad[N_DIMENSION];
	get_gradient(pts[i], grad);
	for (int i = 0; i < N_DIMENSION; i++)
	    loss += grad[i] * grad[i];
    }
    return loss / N_DIMENSION;
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
    vector<int> sparse_rows_left;
    for (int i = 0; i < N_DIMENSION; i++) sparse_rows_left.push_back(i);
    random_shuffle(sparse_rows_left.begin(), sparse_rows_left.end());

    vector<DataPoint> sparse_matrix;

    //Randomize the sparse matrix
    for (int i = 0; i < N_DIMENSION; i++) {
	DataPoint p;
	p.first = sparse_rows_left[i];
	for (int j = 0; j < NUM_SPARSE_ELEMENTS_IN_ROW; j++) {
	    p.second[rand() % N_DIMENSION] = rand() % 100;
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
	for (auto const & x : row) {
	    row[x.first] /= norm_factor;
	}

	norm_row[sparse_matrix[i].first] = norm_factor;
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

    return sparse_matrix;
}

void initialize_model() {

    //Initialize model
    for (int j = 0; j < N_DIMENSION; j++) {
	model[j] = rand() % 100;
	sum_gradient_tilde[j] = 0;
    }
}

void calculate_gradient_tilde(vector<DataPoint> &pts) {

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

int main(void) {
    srand(0);
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(20);
    vector<DataPoint> sparse_matrix = get_sparse_matrix();
    initialize_model();
    random_shuffle(sparse_matrix.begin(), sparse_matrix.end());
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
	calculate_gradient_tilde(sparse_matrix);
	cout << compute_loss(sparse_matrix) << endl;
	vector<int> random_pattern;
	for (int i = 0; i < sparse_matrix.size(); i++) 
	  random_pattern.push_back(i);
	random_shuffle(random_pattern.begin(), random_pattern.end());
	//for (int row = 0; row < sparse_matrix.size(); row++) {
	for (int access = 0; access < random_pattern.size(); access++) {
	    int row = random_pattern[access];
	    DataPoint sparse_row = sparse_matrix[row];
	    double gradient[N_DIMENSION];
	    get_gradient(sparse_row, gradient);

	    for (int i = 0; i < N_DIMENSION; i++) {
		double full_gradient = gradient[i] - gradient_tilde[sparse_row.first][i] + sum_gradient_tilde[i] / N_DIMENSION;
		model[i] -= GAMMA * full_gradient;
	    }
	}
	//for (int j = 0; j < N_DIMENSION; j++) cout << model[j] << endl;
    }
}
