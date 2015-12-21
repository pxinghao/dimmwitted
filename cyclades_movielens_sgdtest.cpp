#include <iostream>
#include "src/util.h"
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <algorithm> 
#include <math.h>
#include <time.h>
#include <map>
#include <unistd.h>
#include "ConnectedComponents/CC_allocation.h"
#include "cyclades.h"

#define N_USERS 943
#define N_RATINGS 100000
#define TOTAL_N_FEATURES 23


#define N_NUMA_NODES 2
#define N_EPOCHS 100
#define BATCH_SIZE 10000
#define NTHREAD 8

using namespace std;

void get_user_data(map<int, vector<double> > &user_dat) {
  ifstream fin("ml-100k/u.user");
  string line;
  while (getline(fin, line)) {
    stringstream linestream(line);
    string occupation, gender;
    double userid, age, zip;
    linestream >> userid >> age >> gender >> occupation >> zip;
    vector<double> features;
    features.push_back(age);
    features.push_back(gender == "M" ? 1 : 0);
    features.push_back(zip);
    user_dat[(int)userid] = features;
  }
  fin.close();
}

void get_item_data(map<int, vector<double> > &item_dat) {
  ifstream fin("ml-100k/u.item");
  string line;
  while (getline(fin, line)) {
    stringstream linestream(line);
    double movie_id, unknown, action, adventure, animation, children, comedy, crime, documentary, drama, fantasy, noir, horror, musical, mystery, romance, scifi, thriller, war, western;
    string movie_title, release_date, video_release_date, url;
    linestream >> movie_id >> movie_title >> release_date >> video_release_date >> url;
    linestream >> unknown >> action >> adventure >> animation >> children >> comedy >> crime >> documentary >> drama >> fantasy >> noir >> horror >> musical >> mystery >> romance >> scifi >> thriller >> war >> western;

    vector<double> features;
    features.push_back(movie_id);
    features.push_back(unknown); 
    features.push_back(action);
    features.push_back(adventure);
    features.push_back(animation);
    features.push_back(children);
    features.push_back(comedy);
    features.push_back(crime);
    features.push_back(documentary);
    features.push_back(drama);
    features.push_back(fantasy);
    features.push_back(noir);
    features.push_back(horror);
    features.push_back(musical);
    features.push_back(mystery);
    features.push_back(romance);
    features.push_back(scifi);
    features.push_back(thriller);
    features.push_back(war);
    features.push_back(western);
    for (int i = 0; i < features.size(); i++) features[i] = rand() % 100;
    item_dat[(int)movie_id] = features;
  }  
  fin.close();
}

void get_full_data(double **mat, double *values) {
  ifstream fin("ml-100k/u.data");
  string line;
  map<int, vector<double> > item_dat, user_dat;
  get_item_data(item_dat);
  get_user_data(user_dat);
  int c = 0;
  while (getline(fin, line)) {  
    stringstream linestream(line);
    int user_id, item_id;
    double rating;
    string timestamp;
    linestream >> user_id >> item_id >> rating >> timestamp;
    vector<double> user_vec = user_dat[user_id];
    vector<double> item_vec = item_dat[item_id];
    values[c] = rating;
    int col = 0;
    for (int i = 0; i < user_vec.size(); i++) {
      mat[c][col] = user_vec[i];
      col++;
    }
    for (int i = 0; i < item_vec.size(); i++) {
      mat[c][col] = item_vec[i];
      col++;
    }
    c++;
  }
  fin.close();
}

double ** extract_data_info(double **dat, vector<int> &p_examples, vector<int> &p_nelems, vector<int> &indices) {
  double ** sparse_mat = (double **)malloc(sizeof(double *) * N_RATINGS);
  int count = 0;
  for (int i = 0; i < N_RATINGS; i++) {
    p_examples.push_back(count);
    int num_non_zero_features = 0;
    for (int j = 0; j < TOTAL_N_FEATURES; j++) {
      if (dat[i][j] != 0) {
	num_non_zero_features += 1;
	indices.push_back(dat[i][j]);
	count += 1;
      }
    }
    int insert_index = 0;
    sparse_mat[i] = (double *)malloc(sizeof(double) * num_non_zero_features);
    for (int j = 0; j < TOTAL_N_FEATURES; j++) {
      if (dat[i][j] != 0) {
	sparse_mat[i][insert_index++] = dat[i][j];
      }
    }
    p_nelems.push_back(num_non_zero_features);
  }
  return sparse_mat;
}

void cyclades_movielens() {

  //Load movie data
  double **mat = (double **)malloc(sizeof(double *) * N_RATINGS);
  for (int i = 0; i < N_RATINGS; i++) mat[i] = (double *)calloc(TOTAL_N_FEATURES, sizeof(double));
  double *values = (double *)malloc(sizeof(double) * N_RATINGS);

  /*for (int i = 0; i < N_RATINGS; i++) {
    for (int j = 0; j < TOTAL_N_FEATURES; j++) {
      mat[i][j] = i+1;
    }
  }
  for (int i = 0; i < N_RATINGS; i++) values[i] = i+1;
  */
  get_full_data(mat, values);
  
  //Extract data info
  vector<int> p_examples, p_nelems, indices;
  double ** sparse_mat = extract_data_info(mat, p_examples, p_nelems, indices);

  //Extract CC info
  int num_batches = (int)ceil(N_RATINGS / (float)BATCH_SIZE);
  //int *numa_aware_indices[NTHREAD*num_batches];
  //int NELEMS[NTHREAD*num_batches];
  int ***numa_aware_indices = new int ** [NTHREAD];
  int **NELEMS = new int *[NTHREAD];
  for (int i = 0; i < NTHREAD; i++) {
    numa_aware_indices[i] = new int * [num_batches];
    NELEMS[i] = (int *)calloc(num_batches, sizeof(int));
  }
  
  Timer t2;
  CC_allocate(mat, N_RATINGS, TOTAL_N_FEATURES, N_NUMA_NODES, numa_aware_indices, NELEMS, BATCH_SIZE, NTHREAD);
  /*cout << "CC ALLOC TIME: " << t2.elapsed() << endl;

  for (int i = 0; i < NTHREAD; i++) {
    cout << "THREAD " << i << endl;
    for (int j = 0; j < num_batches; j++) {
      cout << "BATCH " << j << endl;
      for (int k = 0; k < NELEMS[i][j]; k++) {
	cout << "DATA POINT " << numa_aware_indices[i][j][k] << endl;
      }
    }
    }*/
  
  //Do Cyclades
  Timer t;
  apply_cyclades(sparse_mat, values, p_examples, p_nelems, indices, NELEMS, numa_aware_indices, NTHREAD, N_NUMA_NODES, num_batches, N_EPOCHS);
  cout << "TIME ELAPSED: " << t.elapsed() << endl;
  cout << "LOSS: " << compute_loss(sparse_mat, values, &indices[0], &p_examples[0], p_nelems) << endl;

  //Free mem
  for (int i = 0; i < N_RATINGS; i++) {
    free(mat[i]);
  }
  free(mat);
  free(values);
  for (int i = 0; i < NTHREAD; i++) {
    for (int j = 0; j < num_batches; j++)
      numa_free(numa_aware_indices[i][j], NELEMS[i][j]);
    free(NELEMS[i]);
    free(numa_aware_indices[i]);
  }
  free(NELEMS);
  free(numa_aware_indices);
}

int main(void) {
  setprecision(15);
  for (int i = 0; i < N_RATINGS; i++) models[i] = 0;
  cyclades_movielens();
}
