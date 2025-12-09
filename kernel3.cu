
#include "common.h"
#include "timer.h"

__global__ void gpu3kernel(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d, unsigned int* outputColsPool, unsigned int* numOutputColsPool, float* outputValuesPool) {
  int element = threadIdx.x;
  int row = blockIdx.x;

  unsigned int *outputCols = &outputColsPool[row * csrMatrix2_d->numCols];
  float *outputValues = &outputValuesPool[row * csrMatrix2_d->numCols];
  unsigned int *numOutputCols = &numOutputColsPool[row];

  __shared__ int startIdx;;

  if (row < csrMatrix1_d->numRows) {
    int numElements = csrMatrix1_d->rowPtrs[row+1] - csrMatrix1_d->rowPtrs[row];
    int offset = csrMatrix1_d->rowPtrs[row];
    for (unsigned int i = element; i < numElements; i += blockDim.x) {
        unsigned int col = csrMatrix1_d->colIdxs[offset + i];
        for (unsigned int b_idx = csrMatrix2_d->rowPtrs[col]; b_idx < csrMatrix2_d->rowPtrs[col+1]; b_idx ++) {
            unsigned int col2 = csrMatrix2_d->colIdxs[b_idx];

            float oldVal = atomicAdd(&outputValues[col2], csrMatrix1_d->values[offset + i] * csrMatrix2_d->values[b_idx]);
            if (oldVal == 0.0f) {
                int outputIdx = atomicAdd(numOutputCols, 1);
                outputCols[outputIdx] = col2;
            }
        }
    }

    if (element == 0) {
        startIdx = atomicAdd(&cooMatrix_d->numNonzeros, *numOutputCols);
    }

    __syncthreads();
    for (unsigned int i = element; i < *numOutputCols; i += blockDim.x) {
        cooMatrix_d->values[startIdx + i] = outputValues[outputCols[i]];
        cooMatrix_d->colIdxs[startIdx + i] = outputCols[i];
        cooMatrix_d->rowIdxs[startIdx + i] = row;
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
  
  int threadsPerBlock = 128;
  int numBlocks = csrMatrix1->numRows;
  gpu3kernel<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d, outputCols, numOutputCols, outputValues);

  cudaDeviceSynchronize();

  int numNonzeros;
  cudaMemcpy(&numNonzeros, &cooMatrix_d->numNonzeros, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  printf("num nonzeros: %u", numNonzeros);
}