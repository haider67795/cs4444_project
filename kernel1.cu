
#include "common.h"
#include "timer.h"

__global__ void gpu1kernel_p1(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, unsigned int* outputColsPool, unsigned int* numOutputColsPool, float* outputValuesPool) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= csrMatrix1_d->numNonzeros) return;

  // binary search to find row of element this thread is assigned to
  int low = 0, high = csrMatrix1_d->numRows;
  while (low < high - 1) {
    int mid = (low + high) / 2;
    if (idx < csrMatrix1_d->rowPtrs[mid]) {
      high = mid;
    } else {
      low = mid;
    }
  }
  int row = low;
  
  unsigned int *outputCols = &outputColsPool[row * csrMatrix1_d->numCols];
  float *outputValues = &outputValuesPool[row * csrMatrix1_d->numCols];
  unsigned int *numOutputCols = &numOutputColsPool[row];

  float val = csrMatrix1_d->values[idx];
  int col = csrMatrix1_d->colIdxs[idx];
  
  unsigned int b_start = csrMatrix2_d->rowPtrs[col];
  unsigned int b_end = csrMatrix2_d->rowPtrs[col + 1];

  for (unsigned int b_idx = b_start; b_idx < b_end; ++b_idx) {
    unsigned int col2 = csrMatrix2_d->colIdxs[b_idx];

    float oldVal = atomicAdd(&outputValues[col2], val * csrMatrix2_d->values[b_idx]);
    if (oldVal == 0.0f) {
      int oldOPCs = atomicAdd(numOutputCols, 1);
      outputCols[oldOPCs] = col2;
    }
  }

  
}

__global__ void gpu1kernel_p2(int mat1NumCols, COOMatrix* cooMatrix_d, unsigned int* outputColsPool, unsigned int* numOutputColsPool, float* outputValuesPool) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int *outputCols = &outputColsPool[row * mat1NumCols];
  float *outputValues = &outputValuesPool[row * mat1NumCols];
  unsigned int *numOutputCols = &numOutputColsPool[row];

  for (unsigned int i = 0; i < numOutputCols[row]; i ++) {
    unsigned int col = outputCols[i];
    float value = outputValues[col];
    unsigned int j = atomicAdd(&cooMatrix_d->numNonzeros, 1);
    cooMatrix_d->rowIdxs[j] = row;
    cooMatrix_d->colIdxs[j] = col;
    cooMatrix_d->values[j] = value;
  }
}


void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {

  
  float *outputValues;
  cudaMalloc(&outputValues, csrMatrix1->numRows * csrMatrix1->numCols * sizeof(float));
  cudaMemset(outputValues, 0, csrMatrix1->numRows * csrMatrix1->numCols * sizeof(float));
  
  unsigned int *outputCols;
  cudaMalloc(&outputCols, csrMatrix1->numRows * csrMatrix1->numCols * sizeof(unsigned int));

  unsigned int *numOutputCols;
  cudaMalloc(&numOutputCols, csrMatrix1->numRows * sizeof(unsigned int));
  cudaMemset(numOutputCols, 0, csrMatrix1->numRows * sizeof(unsigned int));
  
  int threadsPerBlock = 1024;
  int numBlocks = (csrMatrix1->numNonzeros + threadsPerBlock - 1) / threadsPerBlock;
  gpu1kernel_p1<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, outputCols, numOutputCols, outputValues);

  cudaDeviceSynchronize();

  numBlocks = (csrMatrix1->numRows + threadsPerBlock - 1) / threadsPerBlock;
  gpu1kernel_p2<<<numBlocks, threadsPerBlock>>>(csrMatrix1->numCols, cooMatrix_d, outputCols, numOutputCols, outputValues);
  

  cudaDeviceSynchronize();

}

