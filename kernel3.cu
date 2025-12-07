
#include "common.h"
#include "timer.h"

__global__ void gpu3kernel_p1(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, unsigned int* outputColsPool, unsigned int* numOutputColsPool, float* outputValuesPool) {
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

__global__ void gpu3kernel_p2(int totalOutputElements, int mat2NumCols, int mat1NumRows, COOMatrix* cooMatrix_d, unsigned int* outputColsPool, unsigned int* numOutputColsPool, float* outputValuesPool) {
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

#define BLOCK_SIZE 1024

__global__ void silly_scan(unsigned int* arr, int length, unsigned int* sum) {
  // serial scan of coarse bit idk
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ unsigned int buff_1[BLOCK_SIZE];
  __shared__ unsigned int buff_2[BLOCK_SIZE];
  unsigned int* srcBuff = buff_1;
  unsigned int* destBuff = buff_2;
  srcBuff[BLOCK_SIZE - 1] = 0;
  for (int step = 0; step < length; step += BLOCK_SIZE) {
    if (length <= step + idx) {return;}
    unsigned int prev = srcBuff[BLOCK_SIZE - 1];
    unsigned int* segment = arr + step; // pointer arithmetic go brrr
    srcBuff[idx] = segment[idx]; // load to shared memory
    destBuff[idx] = 0;
    for (int i = 1; i < BLOCK_SIZE; i <<= 1) {
      if (i <= idx) {
        destBuff[idx] = srcBuff[idx] + srcBuff[idx - i];
      }
      unsigned int* temp = srcBuff;
      srcBuff = destBuff;
      destBuff = srcBuff;
      __syncthreads();
    }
    srcBuff[idx] += prev;
    segment[idx] = srcBuff[idx];
    __syncthreads();
    if (step + idx == length - 1) {
      *sum = srcBuff[idx];
    }
  }
  
}

void spmspm_gpu3(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {

  
  float *outputValues;
  cudaMalloc(&outputValues, csrMatrix1->numRows * csrMatrix2->numCols * sizeof(float));
  cudaMemset(outputValues, 0, csrMatrix1->numRows * csrMatrix2->numCols * sizeof(float));
  
  unsigned int *outputCols;
  cudaMalloc(&outputCols, csrMatrix1->numRows * csrMatrix2->numCols * sizeof(unsigned int));

  unsigned int *numOutputCols;
  cudaMalloc(&numOutputCols, csrMatrix1->numRows * sizeof(unsigned int));
  cudaMemset(numOutputCols, 0, csrMatrix1->numRows * sizeof(unsigned int));
  
  int threadsPerBlock = BLOCK_SIZE;
  int numBlocks = (csrMatrix1->numNonzeros + threadsPerBlock - 1) / threadsPerBlock;
  gpu3kernel_p1<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, outputCols, numOutputCols, outputValues);

  cudaDeviceSynchronize();
  
  unsigned int* totalOutputElements_d;
  cudaMalloc(&totalOutputElements_d, sizeof(unsigned int));

  silly_scan<<<1, threadsPerBlock>>>(numOutputCols, csrMatrix1->numRows, totalOutputElements_d);

  // set numNonzeros of COO outut
  cudaMemcpy(&cooMatrix_d->numNonzeros, totalOutputElements_d, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  
  unsigned int totalOutputElements;
  cudaMemcpy(&totalOutputElements, totalOutputElements_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  numBlocks = (totalOutputElements + threadsPerBlock - 1) / threadsPerBlock;
  gpu3kernel_p2<<<numBlocks, threadsPerBlock>>>(totalOutputElements, csrMatrix2->numCols, csrMatrix1->numRows, cooMatrix_d, outputCols, numOutputCols, outputValues);
  

  cudaDeviceSynchronize();

}