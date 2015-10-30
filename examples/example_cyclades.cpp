#include <vector>
#include <fstream>

#include "Cyclades_sparse_sgd.cpp"

using namespace std;

int loadData(string filename, vector<vector<double> > &data, vector<vector<long> > &dataCols){
  ifstream datafile;
  datafile.open(filename);

  int counter = 0;
  string line;
  while ( getline(datafile, line) ){
    size_t matchPos = -1;
    size_t startPos = -1;
    vector<double> d;
    vector<long> dc;
    long num = 0;
    while (true){
      startPos = matchPos + 1;
      matchPos = line.find('\t', startPos);
      if (matchPos != string::npos){
        num = stoi(line.substr(startPos, matchPos - startPos));
        if (startPos == 0){
          // First number in line; number of features.
          // We do nothing with this for now.
        }else{
          dc.push_back(num);
          d.push_back(drand48());
        }
      }else{
        break;
      }
    }
    data.push_back(d);
    dataCols.push_back(dc);
    counter ++;
  }
  datafile.close();
  return counter;
}

void loadDataAccess(string filename, int numThreads, int numBatches, vector<vector<vector<long> > > &examples){
  ifstream dataAccessFile;
  dataAccessFile.open(filename);

  string line;
  for (int batchID = 0; batchID < numBatches; batchID++){
    vector<vector<long> > examplesByBatch;

    for (int threadID = 0; threadID < numThreads; threadID++){
      vector<long> examplesByBatchThread;

      getline(dataAccessFile, line);

      size_t matchPos = -1;
      size_t startPos = -1;
      long num = 0;

      startPos = matchPos + 1;
      matchPos = line.find('\t', startPos);
      num = stoi(line.substr(startPos, matchPos - startPos));
      // Sanity check
      if (num != batchID){
        cerr << "ERROR: read batchID as " << num << " but expected " << batchID << endl;
      }

      startPos = matchPos + 1;
      matchPos = line.find('\t', startPos);
      num = stoi(line.substr(startPos, matchPos - startPos));
      // Sanity check
      if (num != threadID){
        cerr << "ERROR: read threadID as " << num << " but expected " << threadID << endl;
      }

      // Read examples
      while (true){
        startPos = matchPos + 1;
        matchPos = line.find('\t', startPos);
        if (matchPos != string::npos){
          num = stoi(line.substr(startPos, matchPos - startPos));
          examplesByBatchThread.push_back(num);
        }else{
          break;
        }
      }

      examplesByBatch.push_back(examplesByBatchThread);
    }

    examples.push_back(examplesByBatch);
  }

  dataAccessFile.close();
}

void allocateDataToNUMA(int numBatches, int numThreads, int nfeat, vector<vector<double> > data, vector<vector<long> > dataCols, vector<vector<vector<long> > > examples, vector<SparseVector<double>*>& row_pointers_all, vector<long>& batchNumRows, vector <long>& batchNumCols, vector<long>& batchNumElems){

  long noopDensity = 10;

  for (int ibatch=0; ibatch < numBatches; ibatch++){
    long nrows = 0;
    long ncols = nfeat+1;
    long nelems = 0;

    // Count max number of examples in batch across threads
    int maxCount = 0;
    for (int ithread=0; ithread < numThreads; ithread++){
      if (examples[ibatch][ithread].size() > maxCount){
        maxCount = examples[ibatch][ithread].size();
      }
    }

    int nexp = maxCount * numThreads;

    SparseVector<double>* row_pointers = 
      (SparseVector<double>*) ::operator new(nexp * sizeof(SparseVector<double>));

    for (int ithread=0; ithread < numThreads; ithread++){
      numa_run_on_node(ithread);
      numa_set_localalloc();

      for (int iexp = 0; iexp < examples[ibatch][ithread].size(); iexp++){
        long expID = examples[ibatch][ithread][iexp];
        long length = data[expID].size();
        double* content = new double[length + 1];
        long* cols = new long[length + 1];
        for (int ii = 0; ii < length; ii++){
          content[ii] = data[expID][ii];
          cols[ii]    = dataCols[expID][ii];
          nelems++;
        }
        content[length] = drand48() > 0.8 ? 0 : 1.0;
        cols[length] = nfeat;
        nelems++;
        row_pointers[nrows++] = SparseVector<double>(content, cols, length+1);
      }

      for (int iexp = examples[ibatch][ithread].size(); iexp < maxCount; iexp++){
        long length = 0;
        double* content = new double[length + 1];
        long* cols = new long[length + 1];
        content[length] = drand48() > 0.8 ? 0 : 1.0;
        cols[length] = nfeat;
        nelems++;
        row_pointers[nrows++] = SparseVector<double>(content, cols, length+1);
      }

    }

    // Sanity check that nrows = nexp
    if (nrows != nexp) cout << "nrows = " << nrows << ", nexp = " << nexp << endl;

    row_pointers_all.push_back(row_pointers);
    batchNumRows.push_back(nrows);
    batchNumCols.push_back(ncols);
    batchNumElems.push_back(nelems);
  }

}



/**
 * \brief This is one example of running SGD for logistic regression
 * in DimmWitted. You can find more examples in test/glm_dense.cc
 * and test/glm_sparse.cc, and the documented code in 
 * app/glm_dense_sgd.h
 */
// int main_dataaccess(int argc, char** argv){
int main(int argc, char** argv){
  int numEpochs = 1;
  int numBatches = 100;
  int numThreads = 8;
  int nfeat = 100000;

  vector<vector<double> > data;
  vector<vector<long  > > dataCols;
  vector<vector<vector<long> > > examples;
  vector<long> batchNumRows, batchNumCols, batchNumElems;
  vector<SparseVector<double>*> row_pointers_all;

  loadData("data/dataaccess_data_multinomialLogisticRegression.txt", data, dataCols);
  loadDataAccess("data/dataaccess_nthreads8_multinomialLogisticRegression.txt", numThreads, numBatches, examples);
  allocateDataToNUMA(numBatches, numThreads, nfeat, data, dataCols, examples, row_pointers_all, batchNumRows, batchNumCols, batchNumElems);

  if (false){
    cout << row_pointers_all.size() << " / " << numBatches << endl;
    for (int ibatch = 0; ibatch < numBatches; ibatch++){
      for (int irow = 0; irow < batchNumRows[ibatch]; irow++){
        cout << ibatch << '\t' << irow;
        for (int icol = 0; icol < row_pointers_all[ibatch][irow].n; icol ++){
          cout << "\t(" << row_pointers_all[ibatch][irow].idxs[icol] << ":" << row_pointers_all[ibatch][irow].p[icol] << ")";
        }
        cout << endl;
      }
    }
  }

  double rs = 0.0;
  rs = test_cyc_sparse_sgd<DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING>(numEpochs, numBatches, numThreads, nfeat, row_pointers_all, batchNumRows, batchNumCols, batchNumElems);
  std::cout << "SUM OF MODEL (Should be ~1.3-1.4): " << rs << std::endl;
  return 0;
}

void allocateSyntheticData(int numBatches, int numThreads, int nfeat, int density, int numExamplesPerBatchPerThread, vector<SparseVector<double>*>& row_pointers_all, vector<long>& batchNumRows, vector <long>& batchNumCols, vector<long>& batchNumElems){
  for (int ibatch=0; ibatch < numBatches; ibatch++){
    SparseVector<double>* row_pointers = (SparseVector<double>*) ::operator new(numExamplesPerBatchPerThread * numThreads * sizeof(SparseVector<double>));
    int nrows = 0;
    int nelems = 0;
    for (int ithread = 0; ithread < numThreads; ithread++){
      numa_run_on_node(ithread);
      numa_set_localalloc();
      for (int iexp = 0; iexp < numExamplesPerBatchPerThread; iexp++){
        double* content = new double[density + 1];
        long*   cols    = new long[density + 1];
        for (int idensity = 0; idensity < density; idensity++){
          cols[idensity] = (int)(drand48() * nfeat);
          content[idensity] = drand48();
          nelems ++;
        }
        cols[density] = nfeat;
        content[density] = drand48() > 0.8 ? 0 : 1;
        nelems++;
        row_pointers[nrows++] = SparseVector<double>(content, cols, density+1);
      }
    }
    row_pointers_all.push_back(row_pointers);
    batchNumRows.push_back(nrows);
    batchNumCols.push_back(nfeat+1);
    batchNumElems.push_back(nelems);
  }
}

int main_synthetic(int argc, char** argv){
// int main(int argc, char** argv){
  int numEpochs = 1;
  int numBatches = 100;
  int numExamplesPerBatchPerThread = 200;
  int numThreads = 8;
  int nfeat = 10000;
  int density = 10;

  vector<long> batchNumRows, batchNumCols, batchNumElems;
  vector<SparseVector<double>*> row_pointers_all;

  allocateSyntheticData(numBatches, numThreads, nfeat, density, numExamplesPerBatchPerThread, row_pointers_all, batchNumRows, batchNumCols, batchNumElems);


  double rs = 0.0;
  rs = test_cyc_sparse_sgd<DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING>(numEpochs, numBatches, numThreads, nfeat, row_pointers_all, batchNumRows, batchNumCols, batchNumElems);
  std::cout << "SUM OF MODEL (Should be ~1.3-1.4): " << rs << std::endl;
  return 0;

}