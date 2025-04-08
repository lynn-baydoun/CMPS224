
#include "common.h"

void spmspm_cpu0(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3) {
    //clear output COO matrix
    clearCOOMatrix(cooMatrix3);
    //iterate over rows of A (csrMatrix1)
    for (unsigned int i = 0; i < csrMatrix1->numRows; i++) {
        //iterate over columns of B (cscMatrix2)
        for (unsigned int j = 0; j < cscMatrix2->numCols; j++) {
            float sum = 0.0f;

            //two-pointer technique for dot product
            unsigned int a_ptr = csrMatrix1->rowPtrs[i];
            unsigned int a_end = csrMatrix1->rowPtrs[i + 1];

            unsigned int b_ptr = cscMatrix2->colPtrs[j];
            unsigned int b_end = cscMatrix2->colPtrs[j + 1];

            while (a_ptr < a_end && b_ptr < b_end) {
                unsigned int a_col = csrMatrix1->colIdxs[a_ptr];
                unsigned int b_row = cscMatrix2->rowIdxs[b_ptr];

                if (a_col == b_row) {
                    sum += csrMatrix1->values[a_ptr] * cscMatrix2->values[b_ptr];
                    a_ptr++;
                    b_ptr++;
                } else if (a_col < b_row) {
                    a_ptr++;
                } else {
                    b_ptr++;
                }
            }

            //if the result is non-zero, store in COO format
            if (sum != 0.0f) {
                if (cooMatrix3->numNonzeros >= cooMatrix3->capacity) {
                    printf("COO output matrix capacity exceeded!\n");
                    exit(EXIT_FAILURE);
                }

                cooMatrix3->rowIdxs[cooMatrix3->numNonzeros] = i;
                cooMatrix3->colIdxs[cooMatrix3->numNonzeros] = j;
                cooMatrix3->values[cooMatrix3->numNonzeros] = sum;
                cooMatrix3->numNonzeros++;
            }
        }
    }






}
