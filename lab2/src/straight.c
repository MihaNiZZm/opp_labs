#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "mpi.h"

#define SIZE_OF_VECTOR 5432

double const TAU = 0.00001;
double const EPSILON = 1.0e-5;

void freeMemory(double* v1, double* v2, double* v3, double* v4, double* v5, double* m1) {
    free(v1);
    free(v2);
    free(v3);
    free(v4);
    free(v5);
    free(m1);
}

void setDataFirstVariant(double* matrixA, double* vectorB, double* vectorXn, double* vectorXn1, double* vectorAXn, double* vectorDiffAXnB) {
    for (int i = 0; i < SIZE_OF_VECTOR; ++i) {
        for (int j = 0; j < SIZE_OF_VECTOR; ++j) {
            if (i == j) {
                matrixA[i * SIZE_OF_VECTOR + j] = 2.0;
            }
            else {
                matrixA[i * SIZE_OF_VECTOR + j] = 1.0;
            }
        }
    }

    for (int i = 0; i < SIZE_OF_VECTOR; ++i) {
        vectorB[i] = SIZE_OF_VECTOR + 1.0;
        vectorXn[i] = 0.0;
        vectorXn1[i] = 0.0;
        vectorDiffAXnB[i] = 0.0;
        vectorAXn[i] = 0.0;
    }
}

void printVector(double* vector, int vectorSize) {
    printf("Vector: [");
    for (int i = 0; i < vectorSize - 1; ++i) {
        printf("%.1lf, ", vector[i]);
    }
    printf("%.1lf]\n", vector[vectorSize - 1]);
}

void getDiffOfVectors(const double* v1, const double* v2, double* dstVector, double multiplier, int size) {
    for (int i = 0; i < size; ++i) {
        dstVector[i] = v1[i] - multiplier * v2[i];
    }
}

void getMatrixVectorMultiplication(const double* srcMatrix, const double* srcVector, double* dstVector, int axisSize) {
    for (int i = 0; i < axisSize; ++i) {
        dstVector[i] = 0.0;
        for (int j = 0; j < axisSize; ++j) {
            dstVector[i] += srcMatrix[i * axisSize + j] * srcVector[j];
        }
    }
}

void getValues(const double* srcVector, double* dstVector, int size) {
    for (int i = 0; i < size; ++i) {
        dstVector[i] = srcVector[i];
    }
}

double getSquaredNorm(const double* data, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += data[i] * data[i];
    }
    return result;
}

bool isSolved(const double* matrixA, const double* vectorX, const double* vectorB, double* vectorAXn, double* vectorDiffAXnB, double squaredNormB, double threshold, int axisSize) {
    getMatrixVectorMultiplication(matrixA, vectorX, vectorAXn, axisSize);
    getDiffOfVectors(vectorAXn, vectorB, vectorDiffAXnB, 1.0, axisSize);
    double squaredNormAXnB = getSquaredNorm(vectorDiffAXnB, axisSize);
    double resultNorm = squaredNormAXnB / squaredNormB;
    return (resultNorm < threshold * threshold) ? true : false;
}

void getNextX(double* vectorX, double* vectorDiffAXnB, double* vectorXn1, double multiplier, int axisSize) {
    getDiffOfVectors(vectorX, vectorDiffAXnB, vectorXn1, multiplier, axisSize);
    getValues(vectorXn1, vectorX, axisSize);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int numberOfProcesses, currentProcess;
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    if (currentProcess == 0) {
        double start, end;
	
        double* vectorXn = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
        double* vectorXn1 = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
        double* vectorB = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
        double* vectorAXn = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
        double* vectorDiffAXnB = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
        double* matrixA = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR * SIZE_OF_VECTOR);

        double squaredNormB;

        setDataFirstVariant(matrixA, vectorB, vectorXn, vectorXn1, vectorAXn, vectorDiffAXnB);

        start = MPI_Wtime();

        squaredNormB = getSquaredNorm(vectorB, SIZE_OF_VECTOR);

        while(!(isSolved(matrixA, vectorXn, vectorB, vectorAXn, vectorDiffAXnB, squaredNormB, EPSILON, SIZE_OF_VECTOR))) {
            getNextX(vectorXn, vectorDiffAXnB, vectorXn1, TAU, SIZE_OF_VECTOR);
	}

        end = MPI_Wtime();

        printf("Time spent: %.2lf seconds.\n", end - start);
        //printVector(vectorXn, SIZE_OF_VECTOR);
        freeMemory(vectorXn, vectorXn1, vectorB, vectorAXn, vectorDiffAXnB, matrixA);
    }

    MPI_Finalize();
    
    return 0;
}
