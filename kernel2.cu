
#include "common.h"
#include "timer.h"

__global__ void gpu2kernel_p1(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, unsigned int* outputColsPool, unsigned int* numOutputColsPool, float* outputValuesPool) {
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
  
  unsigned int *outputCols = &outputColsPool[row * csrMatrix2_d->numCols];
  float *outputValues = &outputValuesPool[row * csrMatrix2_d->numCols];
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

__global__ void gpu2kernel_p2(int totalOutputElements, int mat2NumCols, int mat1NumRows, COOMatrix* cooMatrix_d, unsigned int* outputColsPool, unsigned int* numOutputColsPool, float* outputValuesPool) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= totalOutputElements) { return; }

  // binary search to find row of element this element is assigned to
  int low = 0, high = mat1NumRows;
  while (low < high - 1) {
    int mid = (low + high) / 2;
    if (idx < numOutputColsPool[mid]) {
      high = mid;
    } else {
      low = mid;
    }
  }
  int row = low;

  int i = idx - numOutputColsPool[row];
  int col = outputColsPool[row * mat2NumCols + i];
  
  cooMatrix_d->rowIdxs[idx] = row;
  cooMatrix_d->colIdxs[idx] = col;
  cooMatrix_d->values[idx] = outputValuesPool[row * mat2NumCols + col];
}

/*
__global__ void simple_scan(unsigned int* arr, unsigned int* buffer, int length) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (unsigned int i = 1; i < length; i <<= 1) {
    if (i <= idx) {
      arr[i] += 
    }
  }
}
*/

// #include <iostream>

void spmspm_gpu2(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {

  
  float *outputValues;
  cudaMalloc(&outputValues, csrMatrix1->numRows * csrMatrix2->numCols * sizeof(float));
  cudaMemset(outputValues, 0, csrMatrix1->numRows * csrMatrix2->numCols * sizeof(float));
  
  unsigned int *outputCols;
  cudaMalloc(&outputCols, csrMatrix1->numRows * csrMatrix2->numCols * sizeof(unsigned int));

  unsigned int *numOutputCols;
  cudaMalloc(&numOutputCols, csrMatrix1->numRows * sizeof(unsigned int));
  cudaMemset(numOutputCols, 0, csrMatrix1->numRows * sizeof(unsigned int));
  
  int threadsPerBlock = 1024;
  int numBlocks = (csrMatrix1->numNonzeros + threadsPerBlock - 1) / threadsPerBlock;
  gpu2kernel_p1<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, outputCols, numOutputCols, outputValues);

  cudaDeviceSynchronize();
  unsigned int* numOutputCols_h = (unsigned int*) malloc(csrMatrix1->numRows * sizeof(unsigned int));
  
  
  cudaMemcpy(numOutputCols_h, numOutputCols, csrMatrix1->numRows * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  // scan numOutputCols array
  unsigned int totalOutputElements = 0;
  for (int i = 0; i < csrMatrix1->numRows; i++) {
    int temp = numOutputCols_h[i];
    numOutputCols_h[i] = totalOutputElements;
    totalOutputElements += temp;
  }
  cudaMemcpy(numOutputCols, numOutputCols_h, csrMatrix1->numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);

  // set numNonzeros of COO outut matrix
  cudaMemcpy(&cooMatrix_d->numNonzeros, &totalOutputElements, sizeof(unsigned int), cudaMemcpyHostToDevice);
  

  numBlocks = (totalOutputElements + threadsPerBlock - 1) / threadsPerBlock;
  gpu2kernel_p2<<<numBlocks, threadsPerBlock>>>(totalOutputElements, csrMatrix2->numCols, csrMatrix1->numRows, cooMatrix_d, outputCols, numOutputCols, outputValues);
  

  cudaDeviceSynchronize();

}