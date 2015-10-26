// Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef _CYC_SPARSE_SGD_H
#define _CYC_SPARSE_SGD_H

#include "dimmwitted.h"

/**
 * \brief This file shows how to specify the same
 * synthetic model as in app/cyc_dense_sgd.h
 * but store the data as sparse matrix instead
 * of dense matrix.
 *
 * See app/cyc_dense_sgd.h for more detailed 
 * comments.
 */
class CYCModelExample_Sparse{
public:
  double * const p;
  int n;
  
  CYCModelExample_Sparse(int _n):
    n(_n), p(new double[_n]){}

  CYCModelExample_Sparse( const CYCModelExample_Sparse& other ) :
     n(other.n), p(new double[other.n]){
    for(int i=0;i<n;i++){
      p[i] = other.p[i];
    }
  }

};

void f_lr_modelavg(CYCModelExample_Sparse** const p_models, int nreplicas, int ireplica){
  CYCModelExample_Sparse * p_model = p_models[ireplica];
  double l2sum = 0.0;
  for(int i=0;i<p_model->n;i++){
    l2sum = 0.0;
    for(int j=0;j<nreplicas;j++){
      l2sum += p_models[j]->p[i];
    }
    (p_model->p)[i] = l2sum/nreplicas;
  }
}


double f_lr_loss_sparse(const SparseVector<double>* const ex, CYCModelExample_Sparse* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];
  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[ex->idxs[i]];
  }
  return  - label * dot + log(exp(dot) + 1.0);
}

double f_lr_grad_sparse(const SparseVector<double>* const ex, CYCModelExample_Sparse* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];

  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    // std::cout << "(" << ex->idxs[i] << ": " << ex->p[i] << ")\t";
    dot += ex->p[i] * model[ex->idxs[i]];
  }
  // std::cout << std::endl;

  const double d = exp(-dot);
  const double Z = 0.0001 * (-label + 1.0/(1.0+d));

  for(int i=0;i<ex->n-1;i++){
    model[ex->idxs[i]] -= ex->p[i] * Z;
  }

  return 1.0;
}

template<ModelReplType MODELREPL, DataReplType DATAREPL>
double test_cyc_sparse_sgd(long nepoch, long nbatches, long numThreads, long nfeat, std::vector<SparseVector<double>*> row_pointers_all, std::vector<long> batchNumRows, std::vector<long> batchNumCols, std::vector<long> batchNumElems){

  if (false){
    for (int ibatch = 0; ibatch < nbatches; ibatch++){
      for (int irow = 0; irow < batchNumRows[ibatch]; irow++){
        std::cout << ibatch << '\t' << irow;
        for (int icol = 0; icol < row_pointers_all[ibatch][irow].n; icol ++){
          std::cout << "\t(" << row_pointers_all[ibatch][irow].idxs[icol] << ":" << row_pointers_all[ibatch][irow].p[icol] << ")";
        }
        std::cout << std::endl;
      }
    }
  }

  CYCModelExample_Sparse model(nfeat);
  for(int i=0;i<model.n;i++){
    model.p[i] = 0.0;
  }

  // Create DW engines
  SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> ** dwEngines = new SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW>*[nbatches];
  unsigned int * f_handle_grads = new unsigned int[nbatches];
  unsigned int * f_handle_losss = new unsigned int[nbatches];

  for (long ibatch = 0; ibatch < nbatches; ibatch++){
    // Thrid, create a DenseDimmWitted object because the synthetic data set
    // we created is dense. This object has multiple templates,
    //    - double: the type of the data (type of elements in ``examples'')
    //    - CYCModelExample: the type of the model
    //    - MODELREPL: Model replication strategy
    //    - DATAREPL: Data replication strategy
    //    - DW_ROW: Access method
    //
    SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> *
      dw = new SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> (row_pointers_all[ibatch], batchNumRows[ibatch], batchNumCols[ibatch], batchNumElems[ibatch], &model);
    unsigned int f_handle_grad = dw->register_row(f_lr_grad_sparse);
    unsigned int f_handle_loss = dw->register_row(f_lr_loss_sparse);
    dw->register_model_avg(f_handle_grad, f_lr_modelavg);
    dw->register_model_avg(f_handle_loss, f_lr_modelavg);
    dwEngines[ibatch] = dw;
    f_handle_grads[ibatch] = f_handle_grad;
    f_handle_losss[ibatch] = f_handle_loss;
  }

  double l2sum = 0.0;
  for (long iepoch = 0; iepoch < nepoch; iepoch++){
    // std::cout << "Running epoch " << iepoch << "\n";
    double loss = 0.0;
    for (long ibatch = 0; ibatch < nbatches; ibatch++){
      std::cout << "epoch " << iepoch << ",\tbatch " << ibatch << "\n";

      SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> * dw = dwEngines[ibatch];
      dw->exec(f_handle_grads[ibatch]);
      l2sum = 0.0;
      for(int i=0;i<nfeat;i++){
        l2sum += model.p[i];
      }
      std::cout.precision(8);
      std::cout << l2sum << "    loss=" << loss << std::endl;
    }
  }

  return l2sum;
}

/**
 * \brief This is one example of running SGD for logistic regression
 * in DimmWitted. You can find more examples in test/cyc_dense.cc
 * and test/cyc_sparse.cc, and the documented code in 
 * app/cyc_dense_sgd.h
 */
//int main(int argc, char** argv){
//  double rs = test_cyc_sparse_sgd<DW_MODELREPL_PERCORE, DW_DATAREPL_SHARDING>();
//  std::cout << "SUM OF MODEL (Should be ~1.3-1.4): " << rs << std::endl;
//  return 0;
//}

#endif

