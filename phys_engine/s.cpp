    __device__ void matrix_multiply_6x6(float A[6][6], float B[6][6], float C[6][6]) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < 6; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    __device__ void qr_decomposition_6x6(float A[6][6], float Q[6][6], float R[6][6]) {
        float u[6], dot_product, norm;

        // Initialize R to be the same as A initially
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                R[i][j] = A[i][j];
            }
        }

        // QR Decomposition using Gram-Schmidt process
        for (int k = 0; k < 6; k++) {
            // Compute the k-th column of Q
            for (int i = 0; i < 6; i++) {
                u[i] = R[i][k];
            }

            // Orthogonalize u against previous columns of Q
            for (int j = 0; j < k; j++) {
                dot_product = 0.0;
                for (int i = 0; i < 6; i++) {
                    dot_product += Q[i][j] * R[i][k];
                }
                
                for (int i = 0; i < 6; i++) {
                    u[i] -= dot_product * Q[i][j]; //have to divided by norm of Q[:][] and R[:][] 
            }

            // Compute the norm of u
            norm = 0.0;
            for (int i = 0; i < 6; i++) {
                norm += u[i] * u[i];
            }
            norm = sqrtf(norm);

            // Normalize u to produce the k-th column of Q
            if (norm > 1e-6) {  // Check for non-zero norm
                for (int i = 0; i < 6; i++) {
                    Q[i][k] = u[i] / norm;
                }
            } else {
                // If norm is zero, set Q column to zero vector
                for (int i = 0; i < 6; i++) {
                    Q[i][k] = 0.0;
                }
            }

            // Update the k-th column of R
            for (int j = k; j < 6; j++) {
                dot_product = 0.0;
                for (int i = 0; i < 6; i++) {
                    dot_product += Q[i][k] * A[i][j];
                }
                R[k][j] = dot_product;
            }
        }

        // Validate the orthogonality of Q
        bool is_orthogonal = true;
        for (int i = 0; i < 6 && is_orthogonal; i++) {
            for (int j = i + 1; j < 6; j++) {
                dot_product = 0.0;
                for (int k = 0; k < 6; k++) {
                    dot_product += Q[k][i] * Q[k][j];
                }
                if (fabs(dot_product) > 1e-3) {
                    is_orthogonal = false;
                    break;
                }
            }
        }

        // Validate that R is upper triangular
        bool is_upper_triangular = true;
        for (int i = 0; i < 6 && is_upper_triangular; i++) {
            for (int j = 0; j < i; j++) {
                if (fabs(R[i][j]) > 1e-3) {
                    is_upper_triangular = false;
                    printf("R is not upper triangular at (%d, %d), value = %f", i, j, R[i][j]);
                    break;
                }
            }
        }


        if (is_orthogonal && is_upper_triangular) {
            // printf("QR decomposition successful: Q is orthogonal and R is upper triangular.");
        } else {
            if (!is_orthogonal) {
                printf("QR decomposition failed: Q is not orthogonal.");
            }
            if (!is_upper_triangular) {
                printf("QR decomposition failed: R is not upper triangular.");
            }
        }
    }

    __device__ void qr_algorithm_6x6(float A[6][6], float eigenvalues[6], float eigenvectors[6][6]) {
        float Q[6][6], R[6][6], temp[6][6];
        int max_iterations = 1000;
        float tolerance = 1e-6;

        // Initialize eigenvectors as the identity matrix
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // QR Algorithm iterations
        for (int iter = 0; iter < max_iterations; iter++) {
            qr_decomposition_6x6(A, Q, R);
            matrix_multiply_6x6(R, Q, A);

            // Update eigenvectors
            matrix_multiply_6x6(eigenvectors, Q, temp);

            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    eigenvectors[i][j] = temp[i][j];
                }
            }

            // Check for convergence (optional)
            float max_off_diag = 0.0;
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    if (i != j) {
                        max_off_diag = fmaxf(max_off_diag, fabs(A[i][j]));
                    }
                }
            }

            if (max_off_diag < tolerance) {
                break;
            }
        }

        // Extract eigenvalues from the diagonal of A
        for (int i = 0; i < 6; i++) {
            eigenvalues[i] = A[i][i];
        }
    }