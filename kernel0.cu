#include <cuda.h>
#include <stdio.h>
#include "common.h"

__global__ void kernel0(CSRMatrix d_csrA, CSCMatrix d_cscB, unsigned int* d_rows, unsigned int* d_cols, float* d_vals, unsigned int* d_nnzCounter) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= d_csrA.numRows) return;

    for (int idxA = d_csrA.rowPtrs[row]; idxA < d_csrA.rowPtrs[row + 1]; idxA++) {
        int a_col = d_csrA.colIdxs[idxA];
        float a_val = d_csrA.values[idxA];

        for (int idxB = d_cscB.colPtrs[a_col]; idxB < d_cscB.colPtrs[a_col + 1]; idxB++) {
            int b_col = d_cscB.rowIdxs[idxB]; //CSC stores rows as rows
            float b_val = d_cscB.values[idxB];

            //every thread writes to its own index in global memory
            unsigned int index = atomicAdd(d_nnzCounter, 1);
            d_rows[index] = row;
            d_cols[index] = b_col;
            d_vals[index] = a_val * b_val;
        }
    }
}

void spmspm_gpu0(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2, unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {
    //estimate of max possible non-zeros in result
    unsigned int maxNonzeros = numNonzeros1 * numNonzeros2;

    //output COO arrays allocation
    unsigned int* d_rows;
    unsigned int* d_cols;
    float* d_vals;
    unsigned int* d_nnzCounter;

    cudaMalloc(&d_rows, maxNonzeros * sizeof(unsigned int));
    cudaMalloc(&d_cols, maxNonzeros * sizeof(unsigned int));
    cudaMalloc(&d_vals, maxNonzeros * sizeof(float));
    cudaMalloc(&d_nnzCounter, sizeof(unsigned int));
    cudaMemset(d_nnzCounter, 0, sizeof(unsigned int));

    //launch kernel
    int blockSize = 256;
    int numBlocks = (numRows1 + blockSize - 1) / blockSize;

    kernel0<<<numBlocks, blockSize>>>(*csrMatrix1, *cscMatrix2, d_rows, d_cols, d_vals, d_nnzCounter);
    cudaDeviceSynchronize();

    //copy nnz count back
    unsigned int h_nnz;
    cudaMemcpy(&h_nnz, d_nnzCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //output COO matrix prep
    cooMatrix3->numRows = numRows1;
    cooMatrix3->numCols = numCols2;
    cooMatrix3->numNonzeros = h_nnz;
    cooMatrix3->rowIdxs = (unsigned int*)malloc(h_nnz * sizeof(unsigned int));
    cooMatrix3->colIdxs = (unsigned int*)malloc(h_nnz * sizeof(unsigned int));
    cooMatrix3->values = (float*)malloc(h_nnz * sizeof(float));

    cudaMemcpy(cooMatrix3->rowIdxs, d_rows, h_nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix3->colIdxs, d_cols, h_nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix3->values, d_vals, h_nnz * sizeof(float), cudaMemcpyDeviceToHost);

    //free GPU memory
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_nnzCounter);
}
