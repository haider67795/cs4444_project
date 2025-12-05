// spmspm_gpu0.cu
#include "common.h"
#include "matrix.h"

__global__ void gpu0kernel(CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d,
                           COOMatrix* cooMatrix_d) {
  int rowId = blockDim.x * blockIdx.x + threadIdx.x;
  if (rowId >= csrMatrix1_d->numRows) return;

  unsigned int a_start = csrMatrix1_d->rowPtrs[rowId];
  unsigned int a_end = csrMatrix1_d->rowPtrs[rowId + 1];

  for (unsigned int a_idx = a_start; a_idx < a_end; ++a_idx) {
    unsigned int b_row = csrMatrix1_d->colIdxs[a_idx];
    float a_val = csrMatrix1_d->values[a_idx];
    if (b_row >= csrMatrix2_d->numRows) continue;

    unsigned int b_start = csrMatrix2_d->rowPtrs[b_row];
    unsigned int b_end = csrMatrix2_d->rowPtrs[b_row + 1];

    for (unsigned int b_idx = b_start; b_idx < b_end; ++b_idx) {
      unsigned int j = csrMatrix2_d->colIdxs[b_idx];
      float b_val = csrMatrix2_d->values[b_idx];
      float prod = a_val * b_val;

      if (prod != 0.0f) {
        int pos = atomicAdd(&(cooMatrix_d->numNonzeros), 1);
        cooMatrix_d->rowIdxs[pos] = rowId;
        cooMatrix_d->colIdxs[pos] = j;
        cooMatrix_d->values[pos] = prod;
      }
    }
  }
}

void spmspm_gpu0(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2,
                 CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d,
                 COOMatrix* cooMatrix_d) {
  int threadsPerBlock = 1024;
  int numBlocks = (csrMatrix1->numRows + threadsPerBlock - 1) / threadsPerBlock;

  gpu0kernel<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d,
                                             cooMatrix_d);
  cudaDeviceSynchronize();
}
