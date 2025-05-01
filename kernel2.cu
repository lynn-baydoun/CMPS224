#include <cuda.h>
#include <stdio.h>
#include "common.h"

#define TILE_SIZE 32
#define MAX_COLS_PER_THREAD 128

__global__ void kernel2(CSRMatrix d_csrA, CSCMatrix d_cscB, unsigned int* d_rows, unsigned int* d_cols, float* d_vals, unsigned int* d_nnzCounter, int numColsB) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_csrA.numRows) return;
    
    //shared memory for storing tiles of matrix B
    __shared__ int s_bColIndices[TILE_SIZE];
    __shared__ float s_bValues[TILE_SIZE];
    __shared__ int s_tileStart, s_tileEnd;
    
    float acc[MAX_COLS_PER_THREAD] = {0};
    
    for (int idxA = d_csrA.rowPtrs[row]; idxA < d_csrA.rowPtrs[row + 1]; idxA++) {
        int a_col = d_csrA.colIdxs[idxA];
        float a_val = d_csrA.values[idxA];
        
        //process column a_col of matrix B in tiles
        for (int tileStart = d_cscB.colPtrs[a_col]; 
                tileStart < d_cscB.colPtrs[a_col + 1]; 
                tileStart += TILE_SIZE) {
            
                int tileEnd = min(tileStart + TILE_SIZE, d_cscB.colPtrs[a_col + 1]);
                
                //first thread in warp loads tile size info
                if (threadIdx.x == 0) {
                    s_tileStart = tileStart;
                    s_tileEnd = tileEnd;
                }
                __syncthreads();
                
                //each thread in the block loads one or more elements
                for (int i = threadIdx.x; i < (s_tileEnd - s_tileStart); i += blockDim.x) {
                    int idx = s_tileStart + i;
                    s_bColIndices[i] = d_cscB.rowIdxs[idx];
                    s_bValues[i] = d_cscB.values[idx];
                }
                __syncthreads();
                
                //process loaded tile
                for (int i = 0; i < (s_tileEnd - s_tileStart); i++) {
                    int b_col = s_bColIndices[i];
                    float b_val = s_bValues[i];
                    if (b_col < MAX_COLS_PER_THREAD) {
                        acc[b_col] += a_val * b_val;
                    }
                }
                __syncthreads();
            }
    }
    
    for (int col = 0; col < min(numColsB, MAX_COLS_PER_THREAD); col++) {
        if (acc[col] != 0.0f) {
            unsigned int index = atomicAdd(d_nnzCounter, 1);
            d_rows[index] = row;
            d_cols[index] = col;
            d_vals[index] = acc[col];
        }
    }
    
}

//handle case where number of columns exceeds MAX_COLS_PER_THREAD
__global__ void kernel2_large(CSRMatrix d_csrA, CSCMatrix d_cscB, unsigned int* d_rows, unsigned int* d_cols, float* d_vals, unsigned int* d_nnzCounter, int numColsB, int colBlockStart) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_csrA.numRows) return;
    
    __shared__ int s_bColIndices[TILE_SIZE];
    __shared__ float s_bValues[TILE_SIZE];
    __shared__ int s_tileStart, s_tileEnd;
    
    float acc[MAX_COLS_PER_THREAD] = {0};
    
    for (int idxA = d_csrA.rowPtrs[row]; idxA < d_csrA.rowPtrs[row + 1]; idxA++) {
        int a_col = d_csrA.colIdxs[idxA];
        float a_val = d_csrA.values[idxA];
        
        for (int tileStart = d_cscB.colPtrs[a_col]; tileStart < d_cscB.colPtrs[a_col + 1]; tileStart += TILE_SIZE) {
            
            int tileEnd = min(tileStart + TILE_SIZE, d_cscB.colPtrs[a_col + 1]);
            
            if (threadIdx.x == 0) {
                s_tileStart = tileStart;
                s_tileEnd = tileEnd;
            }
            __syncthreads();
            
            for (int i = threadIdx.x; i < (s_tileEnd - s_tileStart); i += blockDim.x) {
                int idx = s_tileStart + i;
                s_bColIndices[i] = d_cscB.rowIdxs[idx];
                s_bValues[i] = d_cscB.values[idx];
            }
            __syncthreads();
            
            for (int i = 0; i < (s_tileEnd - s_tileStart); i++) {
                int b_col = s_bColIndices[i];
                
                //skip if column is not in our target block
                if (b_col < colBlockStart || b_col >= colBlockStart + MAX_COLS_PER_THREAD) {
                    continue;
                }
                
                float b_val = s_bValues[i];
                acc[b_col - colBlockStart] += a_val * b_val;
            }
            __syncthreads();
        }
    }
    
    for (int localCol = 0; localCol < MAX_COLS_PER_THREAD; localCol++) {
        int col = colBlockStart + localCol;
        if (col < numColsB && acc[localCol] != 0.0f) {
            unsigned int index = atomicAdd(d_nnzCounter, 1);
            d_rows[index] = row;
            d_cols[index] = col;
            d_vals[index] = acc[localCol];
        }
    }
    
}

void spmspm_gpu2(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2, unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {
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
    
    int blockSize = 256;
    int numBlocks = (numRows1 + blockSize - 1) / blockSize;
    
    if (numCols2 <= MAX_COLS_PER_THREAD) {
        kernel2<<<numBlocks, blockSize>>>(*csrMatrix1, *cscMatrix2, d_rows, d_cols, d_vals, d_nnzCounter, numCols2);
        cudaDeviceSynchronize();
    } else {
        //for larger matrices, process columns in blocks
        for (int colBlock = 0; colBlock < numCols2; colBlock += MAX_COLS_PER_THREAD) {
            kernel2_large<<<numBlocks, blockSize>>>(*csrMatrix1, *cscMatrix2, d_rows, d_cols, d_vals, d_nnzCounter, numCols2, colBlock);
            cudaDeviceSynchronize();
        }
    }
    
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