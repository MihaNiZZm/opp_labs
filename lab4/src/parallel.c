#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define NUMBER_OF_DIMENSIONS 2
#define ZERO_PROCESS 0

#define N1 4200
#define N2 3600
#define N3 3000

#define P1 4
#define P2 3

#define MATRIX_B_SPREAD_TAG 100
#define MATRIX_C_SEND_TAG 200

void multiply(int *partA, int *partB, int *partC, const int sizeOfPartA, const int sizeOfPartB, const int n2){
    for (int i = 0; i < sizeOfPartA; i++) {
        for (int j = 0; j < sizeOfPartB; j++) {
            partC[i * sizeOfPartB + j] = 0;
            for (int k = 0; k < n2; k++) {
                partC[i * sizeOfPartB + j] += partA[i * n2 + k] * partB[k * sizeOfPartB + j];
            }
        }
    }
}

void setA(int* a, const int n1, const int n2) {
    for (int i = 0; i < n1 * n2; ++i) {
        a[i] = 65;
    }
}

void setB(int* a, const int n2, const int n3) {
    for (int i = 0; i < n2 * n3; ++i) {
        a[i] = 66;
    }
}

void printMatrix(int dim1, int dim2, int* matrix) {
    for (int i = 0; i < dim1 * dim2; ++i) {
        printf("%d ", matrix[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int currentProcess, numberOfProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    MPI_Comm gridComm, rowsComm, colsComm;

    const int dimensions[] = { P1, P2 };
    const int periodic[] = { 0, 0 };
    const int reorder = 0;
    const int rowsRemain[] = { 1, 0 };
    const int colsRemain[] = { 0, 1 };

    MPI_Cart_create(MPI_COMM_WORLD, NUMBER_OF_DIMENSIONS, dimensions, periodic, reorder, &gridComm);
    MPI_Cart_sub(gridComm, rowsRemain, &rowsComm);
    MPI_Cart_sub(gridComm, colsRemain, &colsComm);

    int coordinates[NUMBER_OF_DIMENSIONS];
    MPI_Cart_coords(gridComm, currentProcess, NUMBER_OF_DIMENSIONS, coordinates);

    int rowsRank, rowsSize;
    MPI_Comm_rank(rowsComm, &rowsRank);
    MPI_Comm_size(rowsComm, &rowsSize);

    int colsRank, colsSize;
    MPI_Comm_rank(colsComm, &colsRank);
    MPI_Comm_size(colsComm, &colsSize);
    MPI_Status status;

    int* A;
    int* B;
    int* C;

    int sizeOfPartA = N1 / P1;
    int sizeOfPartB = N3 / P2;

    if (currentProcess == ZERO_PROCESS) {
        A = (int*) malloc(sizeof(int) * N1 * N2);
        B = (int*) malloc(sizeof(int) * N2 * N3);
        C = (int*) malloc(sizeof(int) * N1 * N3);

        setA(A, N1, N2);
        setB(B, N2, N3);
    }

    int *partA = (int*) malloc(sizeof(int) * sizeOfPartA * N2);
    int *partB = (int*) malloc(sizeof(int) * sizeOfPartB * N2);
    int *partC = (int*) malloc(sizeof(int) * sizeOfPartA * sizeOfPartB);

    double start, end;
    if (currentProcess == ZERO_PROCESS) {
        start = MPI_Wtime();
    }

    if (colsRank == ZERO_PROCESS) {
        // Рассылаем кусочки матрицы А по строчкам.
        MPI_Scatter(A, sizeOfPartA * N2, MPI_INT, partA, sizeOfPartA * N2, MPI_INT, ZERO_PROCESS, rowsComm);
        // 1 - source buffer, 2 - размер передаваемого куска данных, 3 - тип, 4 - dest buffer, 5 - размер принимаемого куска данных, 6 - тип, 7 - на какой процесс шлем данные, 8 - коммуникатор.
    }

    MPI_Datatype B_SEND;
    // count of blocks, blocklen, stride between beginning of blocks, oldtype, newtype 
    MPI_Type_vector(N2, sizeOfPartB, N3, MPI_INT, &B_SEND);
    MPI_Type_commit(&B_SEND);

    if (currentProcess == 0) {
        int c = 0;
        for (int i = 0; i < N2; ++i) {
            for (int j = 0; j < sizeOfPartB;  ++j) {
                partB[c] = B[i * N3 + j]; 
                ++c;
            }
        }

        for (int i = 1; i < colsSize; ++i){
            // buffer, count, datatype, dest process, tag, communicator
            MPI_Send(&B[i * sizeOfPartB], 1, B_SEND, i, MATRIX_B_SPREAD_TAG, colsComm);
        }
    } else if (coordinates[0] == 0) {
        MPI_Recv(partB, sizeOfPartB * N2, MPI_INT, ZERO_PROCESS, MATRIX_B_SPREAD_TAG, colsComm, &status);
    }

    MPI_Bcast(partA, sizeOfPartA * N2, MPI_INT, ZERO_PROCESS, colsComm); // Рассылаем части матрицы А ненулевым столбцам.
    MPI_Bcast(partB, N2 * sizeOfPartB, MPI_INT, ZERO_PROCESS, rowsComm); // Рассылаем части матрицы B ненулевым строкам.

    multiply(partA, partB, partC, sizeOfPartA, sizeOfPartB, N2);

    MPI_Datatype C_RECV;
    MPI_Type_vector(sizeOfPartA, sizeOfPartB, N3, MPI_INT, &C_RECV);
    MPI_Type_commit(&C_RECV);

    if (currentProcess != ZERO_PROCESS) {
        MPI_Send(partC, sizeOfPartA * sizeOfPartB, MPI_INT, ZERO_PROCESS, MATRIX_C_SEND_TAG, gridComm);
    } else {
        for (int i = 0; i < sizeOfPartA; ++i) {
            for (int j = 0; j < sizeOfPartB;  ++j) {
                C[i * N3 + j] = partC[i * sizeOfPartB + j];
            }
        }

        for (int rank = 1; rank < numberOfProcesses; ++rank) {
            MPI_Recv(partC, sizeOfPartA * sizeOfPartB, MPI_INT, rank, MATRIX_C_SEND_TAG, gridComm, &status);

            int rowEntryPoint = (rank / P2) * sizeOfPartA;
            int colEntryPoint = (rank % P2) * sizeOfPartB;

            for (int i = 0; i < sizeOfPartA; ++i) {
                for (int j = 0; j < sizeOfPartB; ++j) {
                    C[(rowEntryPoint + i) * N3 + colEntryPoint + j] = partC[i * sizeOfPartB + j];
                }
            }
        }

        end = MPI_Wtime();
        printf("Time taken: %f seconds\n", end - start);

        // printMatrix(N1, N3, C);

        free(A);
        free(B);
        free(C);
    }

    free(partA);
    free(partB);
    free(partC);

    MPI_Finalize();
    return 0;
}