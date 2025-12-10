# Baseline (kernel0)

The baseline GPU implementation performs sparse–sparse matrix multiplication by assigning each thread to compute one full row of the output matrix. Since both input matrices are stored in CSR format, each thread begins by reading the nonzeros in its assigned row of A. For every nonzero A[row, k], the thread goes through the entire row k of B and accumulates all partial products into a non-compressed buffer that represents its whole output row. When a column of this buffer is written to for the first time, the thread records that column index in a different buffer so it can later output only the nonzero entries.

After the thread finishes processing all nonzeros in its row, it transfers the nonzero values in the non-compressed row buffer to the COO output matrix by iterating over the column index buffer. This approach is simple and does not require coordination between threads because each thread works independently on a different row. An atomic counter is needed in order to write the COO entries. The baseline leaves room for improvement in how the work is divided across threads and in how the final COO matrix is constructed.

---

# Optimization 1 (kernel1)

This version keeps the same overall sparse–sparse multiplication logic as the baseline, but it changes how the work is divided across threads to expose more parallelism. Instead of assigning one thread to process an entire row of A, kernel1 assigns one thread to each nonzero entry of A. Because CSR does not directly tell us which row a given nonzero belongs to, each thread does a quick binary search on the row pointer array to find its row index. Once the row is known, the thread multiplies its A[row, k] value with the corresponding row k of B and adds the partial results into the same type of per-row non-compressed buffer used in the baseline. Since many threads may now contribute to the same output row, these updates use atomic operations to safely accumulate values and record which columns become nonzero.

After this accumulation phase, a second kernel launches one thread per row to finish the work. Each thread looks at the columns that were updated for its row and writes those (row, col, value) triples into the output COO matrix. This two-step approach does the same math as the baseline, but it splits the work into much smaller pieces. Instead of giving each thread a whole row to process, kernel1 gives each thread a single nonzero from A, which creates many more threads and lets the GPU run a lot more work in parallel. The first kernel handles the partial products, and the second kernel constructs the outut COO matrix from the non-compressed buffers. This parallelism leads to better overall GPU utilization and reduced execution time compared to the baseline.

---

# Optimization 2 (kernel2)

The kernel2 version builds on the kernel1 approach but focuses on making the construction of the output COO matrix more efficient. In kernel1, each output row is written using atomic operations to append entries into the COO matrix one by one, which creates contention on a single global counter and limits performance. kernel2 removes this bottleneck by performing an exclusive scan on the host over the array that holds the number of nonzeros in each row. This scan converts the per-row counts into per-row output offsets, giving each row a fixed, non-overlapping range in the COO arrays. With these offsets in place, the second kernel can launch one thread per output element. In order to determine row, threads in the second kernel binary searche through the recently scanned output offsets. They are then able to write each (row, col, value) triple directly into its correct position in the output COO matrix without using any atomics. This change eliminates the hardware serialization that occurred with the previous versions while constructing the output COO matrix. We do notice a speedup here as well, and kernel2 is consistently faster than kernel1 due to this improvement. However, the gains becomes less noticeable as the matrices get denser.

---

# Optimization 3 (kernel3)

Kernel3 is a simplified version of kernel2 that is optimized for the problem specifications of this assignment. The two separate kernels used in kernel2 are combined into a single kernel in kernel3. Instead of launching one kernel to process every nonzero of A and another to write the COO output, kernel3 assigns one thread block to each row of A, and the threads in that block work together to handle all the nonzeros in that row. This makes it straightforward to manage both the partial products and the COO output for each row without extra kernel launches or binary searches.

A key feature of kernel3 is the single atomicAdd used to update the COO matrix’s numNonzeros value. Each block calls atomicAdd only once to reserve a contiguous region of the output arrays, and then the threads in the block write their COO entries directly into that reserved section. This keeps the structure simple and efficient for the sparse matrices used in this project. Kernel3 is the best option for this assignment because we focus specifically on sparse kernels, but as matrices become much denser, kernel2 becomes more effective due to its more evenly distributed work across many threads.

---

# Performance Results with Speedups

All runtime values are measured in milliseconds (ms). For each GPU + matrix combination we ran 10 trials, with 5 counting as warmup.

Matrix densities tested:

- **Normal:** 10,240 × 10,240 matrix with 327,680 total nonzeros (32 NNZ per row)  
- **Medium:** 10,240 × 10,240 matrix with 1,048,576 total nonzeros (128 NNZ per row)  
- **High:** 10,000 × 10,000 matrix with 10,240,000 total nonzeros (1,024 NNZ per row)

## Normal Density (very sparse)

| GPU Model                    | CPU      | GPU0    | Speedup0 | GPU1    | Speedup1 | GPU2    | Speedup2 | GPU3    | Speedup3 |
|------------------------------|----------|---------|----------|---------|----------|---------|----------|---------|----------|
| RTX 4000 Ada (srv04)         | 99.173   | 14.238  | 6.96×    | 12.816  | 7.74×    | 10.925  | 9.08×    | 6.661   | 14.89×   |
| Quadro RTX 4000 (srv10)      | 96.017   | 30.777  | 3.12×    | 22.538  | 4.26×    | 17.475  | 5.49×    | 14.586  | 6.58×    |
| A100 (40 GB) (slurm)         | 1937.134 | 63.941  | 30.30×   | 55.527  | 34.89×   | 16.248  | 119.22×  | 15.942  | 121.50×  |
| H100 NVL (slurm)             | 85.774   | 7.823   | 10.96×   | 4.998   | 17.16×   | 3.164   | 27.12×   | 2.724   | 31.49×   |


## Medium Density (sparse)

| GPU Model                    | CPU        | GPU0    | Speedup0 | GPU1    | Speedup1 | GPU2    | Speedup2 | GPU3     | Speedup3 |
|------------------------------|------------|---------|----------|---------|----------|---------|----------|----------|----------|
| RTX 4000 Ada (srv04)         | 1498.147   | 167.766 | 8.93×    | 77.552  | 19.32×   | 48.763  | 30.72×   | 36.240   | 41.33×   |
| Quadro RTX 4000 (srv10)      | 1311.741   | 365.201 | 3.59×    | 277.599 | 4.73×    | 234.752 | 5.59×    | 245.746  | 5.34×    |
| A100 (40 GB) (slurm)         | 23150.184  | 361.820 | 63.98×   | 281.727 | 82.17×   | 51.213  | 452.04×  | 44.688   | 517.96×  |
| H100 NVL (slurm)             | 1359.747   | 73.469  | 18.51×   | 33.982  | 40.01×   | 20.306  | 66.96×   | 11.541   | 117.87×  |


## High Density (moderately dense)

| GPU Model                    | CPU         | GPU0     | Speedup0 | GPU1     | Speedup1 | GPU2     | Speedup2 | GPU3      | Speedup3 |
|------------------------------|-------------|----------|----------|----------|----------|----------|----------|-----------|----------|
| RTX 4000 Ada (srv04)         | 16198.833   | 8310.489 | 1.95×    | 1477.293 | 10.97×   | 1443.706 | 11.22×   | 2372.053  | 6.83×    |
| Quadro RTX 4000 (srv10)      | 13580.889   | 16504.342| 0.82×    | 12118.879| 1.12×    | 12044.634| 1.13×    | 15796.518 | 0.86×    |
| A100 (40 GB) (slurm)         | 503176.315  | 4469.840 | 112.57×  | 1858.462 | 270.75×  | 1633.064 | 308.12×  | 2502.300  | 201.11×  |
| H100 NVL (slurm)             | 12264.641   | 3102.731 | 3.95×    | 548.288  | 22.37×   | 542.678  | 22.60×   | 816.872   | 15.01×   |