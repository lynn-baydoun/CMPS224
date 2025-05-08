#include <cuda.h>
#include <stdio.h>
#include "common.h"

#define WARP_SIZE 32
#define HASH_SIZE 64
#define MAX_HASH_ATTEMPTS 64  //to prevent infinite loops

__device__ __forceinline__ unsigned int hash(int key) {
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key % HASH_SIZE;
}

__global__ void kernel3(CSRMatrix d_csrA, CSCMatrix d_cscB, unsigned int* d_rows, unsigned int* d_cols, float* d_vals, unsigned int* d_nnzCounter) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    if (warpId >= d_csrA.numRows) return;

    __shared__ int hashKeys[HASH_SIZE * WARP_SIZE];
    __shared__ float hashVals[HASH_SIZE * WARP_SIZE];

    int* keys = &hashKeys[WARP_SIZE * laneId];
    float* vals = &hashVals[WARP_SIZE * laneId];

    for (int i = laneId; i < HASH_SIZE; i += WARP_SIZE) {
        keys[i] = -1;
        vals[i] = 0.0f;
    }

    __syncwarp();

    int row = warpId;

    for (int idxA = d_csrA.rowPtrs[row]; idxA < d_csrA.rowPtrs[row + 1]; ++idxA) {
        int a_col = d_csrA.colIdxs[idxA];
        float a_val = d_csrA.values[idxA];
        
        //value is too small
        if (fabsf(a_val) < 1e-10f) continue;

        for (int idxB = d_cscB.colPtrs[a_col]; idxB < d_cscB.colPtrs[a_col + 1]; ++idxB) {
            int b_col = d_cscB.rowIdxs[idxB];
            float b_val = d_cscB.values[idxB];
            float product = a_val * b_val;
            
            //product is too small
            if (fabsf(product) < 1e-10f) continue;

            int h = hash(b_col);
            int attempts = 0;
            
            while (attempts < MAX_HASH_ATTEMPTS) {
                if (keys[h] == b_col) {
                    vals[h] += product;
                    break;
                } else if (keys[h] == -1) {
                    if (atomicCAS(&keys[h], -1, b_col) == -1) {
                        vals[h] = product;
                        break;
                    }
                }
                attempts++;
                h = (h + attempts) % HASH_SIZE;
            }
        }
    }

    __syncwarp();

    for (int i = laneId; i < HASH_SIZE; i += WARP_SIZE) {
        if (keys[i] != -1 && vals[i] != 0.0f) {
            unsigned int idx = atomicAdd(d_nnzCounter, 1);
            d_rows[idx] = row;
            d_cols[idx] = keys[i];
            d_vals[idx] = vals[i];
        }
    }
}

void spmspm_gpu3(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2, unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {
    unsigned int maxNonzeros = numNonzeros1 * numNonzeros2;  
    unsigned int* d_rows;
    unsigned int* d_cols;
    float* d_vals;
    unsigned int* d_nnzCounter;

    cudaMalloc(&d_rows, maxNonzeros * sizeof(unsigned int));
    cudaMalloc(&d_cols, maxNonzeros * sizeof(unsigned int));
    cudaMalloc(&d_vals, maxNonzeros * sizeof(float));
    cudaMalloc(&d_nnzCounter, sizeof(unsigned int));
    cudaMemset(d_nnzCounter, 0, sizeof(unsigned int));

    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    const int numWarps = (csrMatrix1->numRows + warpsPerBlock - 1) / warpsPerBlock;
    const int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    kernel3<<<numBlocks, threadsPerBlock>>>(*csrMatrix1, *cscMatrix2, d_rows, d_cols, d_vals, d_nnzCounter);
    cudaDeviceSynchronize();

    unsigned int h_nnz;
    cudaMemcpy(&h_nnz, d_nnzCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cooMatrix3->numRows = numRows1;
    cooMatrix3->numCols = numCols2;
    cooMatrix3->numNonzeros = h_nnz;
    cudaMallocHost(&cooMatrix3->rowIdxs, h_nnz * sizeof(unsigned int));
    cudaMallocHost(&cooMatrix3->colIdxs, h_nnz * sizeof(unsigned int));
    cudaMallocHost(&cooMatrix3->values, h_nnz * sizeof(float));

    cudaMemcpy(cooMatrix3->rowIdxs, d_rows, h_nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix3->colIdxs, d_cols, h_nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix3->values, d_vals, h_nnz * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_nnzCounter);
}