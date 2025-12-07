# Baseline (kernel0)

The baseline GPU implementation performs sparse–sparse matrix multiplication by assigning one thread to compute one full row of the output matrix. Since both input matrices are stored in CSR format, each thread begins by reading the nonzeros in its assigned row of A. For every nonzero A[row, k], the thread goes through the entire row k of B and accumulates all partial products into its own temporary dense buffer that represents the whole output row. Even though the final result is sparse, this buffer is sized for all columns, and each thread updates entries in it as needed. When a column is written for the first time, the thread records that column index so it can later output only the nonzero entries.

After the thread finishes processing all nonzeros in its row, it converts the dense buffer into COO format by writing out only the columns that were marked as nonzero. This approach is simple and does not require coordination between threads during the accumulation phase, because each thread works independently on a different row. The only synchronization needed is the atomic counter used when writing the final COO entries. However, this method still uses a large amount of temporary memory, since every row gets a full dense buffer regardless of how many nonzeros it actually produces. The baseline leaves room for improvement in how the work is divided across threads and in how the final COO entries are written out.

---

# Optimization 1 (kernel1)

This version keeps the same overall sparse–sparse multiplication logic as the baseline, but it changes how the work is divided across threads to expose more parallelism. Instead of assigning one thread to process an entire row of A, kernel1 assigns one thread to each nonzero entry of A. Because CSR does not directly tell us which row a given nonzero belongs to, each thread does a quick binary search on the row pointer array to find its row index. Once the row is known, the thread multiplies its A[row, k] value with the corresponding row k of B and adds the partial results into the same type of per-row dense buffer used in the baseline. Since many threads may now contribute to the same output row, these updates use atomic operations to safely accumulate values and record which columns become nonzero.

After this accumulation phase, a second kernel launches one thread per row to finish the work. Each thread looks at the columns that were updated for its row and writes those (row, col, value) triples into the COO output. This two-step approach does the same math as the baseline, but it splits the work into much smaller pieces. Instead of giving each thread a whole row to process, kernel1 gives each thread a single nonzero from A, which creates many more threads and lets the GPU run a lot more work in parallel. The first kernel handles the partial products, and the second kernel writes out the final nonzero entries for each row. This parallelism leads to better overall GPU utilization compared to the baseline and speeds up execution time by a lot compared to the baseline.

---

# Optimization 2 (kernel2)

The kernel2 version builds on the kernel1 approach but focuses on making the final output step more efficient. In kernel1, each output row is written using atomic operations to append entries into the COO matrix one by one, which creates contention on a single global counter and limits performance. kernel2 removes this bottleneck by performing an exclusive scan over the number of nonzeros in each row. This scan converts the per-row counts into per-row output offsets, giving each row a fixed, non-overlapping range in the COO arrays. With these offsets in place, the second kernel can launch one thread per output element and write each (row, col, value) triple directly into its correct position without using any atomics. This change greatly reduces synchronization overhead and makes the writeback phase much more efficient. We do notice a speedup here as well, and kernel2 is consistently faster because of the more efficient writeback step. However, the improvement becomes less noticeable as the matrices get denser.


# Performance Results with Speedups

All runtime values are measured in milliseconds (ms).

Matrix densities tested:

- **Normal:** 10,240 × 10,240 matrix with 327,680 total nonzeros (32 NNZ per row)  
- **Medium:** 10,240 × 10,240 matrix with 1,048,576 total nonzeros (128 NNZ per row)  
- **High:** 10,000 × 10,000 matrix with 10,240,000 total nonzeros (1,024 NNZ per row)


## Normal Density (very sparse)

| GPU Model               | CPU   | Kernel0 | Speedup0 | Kernel1 | Speedup1 | Kernel2 | Speedup2 |
|-------------------------|-------|---------|----------|---------|----------|---------|----------|
| RTX 4000 Ada (srv04)    | 99.17 | 14.24   | 6.96×    | 12.82   | 7.74×    | 10.93   | 9.08×    |
| Quadro RTX 4000 (srv10) | 96.02 | 30.78   | 3.12×    | 22.54   | 4.26×    | 17.48   | 5.49×    |


## Medium Density (sparse)

| GPU Model               | CPU     | Kernel0 | Speedup0 | Kernel1 | Speedup1 | Kernel2 | Speedup2 |
|-------------------------|---------|---------|----------|---------|----------|---------|----------|
| RTX 4000 Ada (srv04)    | 1498.15 | 167.77  | 8.93×    | 77.55   | 19.32×   | 48.76   | 30.72×   |
| Quadro RTX 4000 (srv10) | 1311.74 | 365.20  | 3.59×    | 277.60  | 4.72×    | 234.75  | 5.59×    |


## High Density (moderately dense)

| GPU Model               | CPU      | Kernel0  | Speedup0 | Kernel1  | Speedup1 | Kernel2  | Speedup2 |
|-------------------------|----------|----------|----------|----------|----------|----------|----------|
| RTX 4000 Ada (srv04)    | 16198.83 | 8310.49  | 1.95×    | 1477.29  | 10.97×   | 1443.71  | 11.22×   |
| Quadro RTX 4000 (srv10) | 13580.89 | 16504.34 | 0.82×    | 12118.88 | 1.12×    | 12044.63 | 1.13×    |



