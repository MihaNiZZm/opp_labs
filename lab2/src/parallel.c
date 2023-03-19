#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi/mpi.h>

#define SIZE_OF_VECTOR 5432
#define ZERO_PROCESS 0

double const TAU = 0.00001;
double const EPSILON = 1.0e-5;

void deleteDouble(double* data) {
    if (data) {
        free(data);
    }
}

void deleteInt(int* data) {
    if (data) {
        free(data);
    }
}

void freeMemory(double* e1, double* e2, double* e3, double* e4, double* e5, double* e6, double* e7, double* e8, double* e9, double* e10, int* e11, int* e12, int* e13, int* e14) {
    deleteDouble(e1);
    deleteDouble(e2);
    deleteDouble(e3);
    deleteDouble(e4);
    deleteDouble(e5);
    deleteDouble(e6);
    deleteDouble(e7);
    deleteDouble(e8);
    deleteDouble(e9);
    deleteDouble(e10);

    deleteInt(e11);
    deleteInt(e12);
    deleteInt(e13);
    deleteInt(e14);
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

void printVector(const double* vector, int vectorSize) {
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
        for (int j = 0; j < SIZE_OF_VECTOR; ++j) {
            dstVector[i] += srcMatrix[i * SIZE_OF_VECTOR + j] * srcVector[j];
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

int getShift(int* sizesOfParts, int currentProcess) {
    int res = 0;
    for (int i = 0; i < currentProcess; ++i) {
        res += sizesOfParts[i];
    }
    return res;
}

void calculateSquaredNormOfVector(double* dstSquaredNorm, const double* srcVector, int* sizesOfPartsOfVector, int* displacementsOfPartsOfVector, int currentProcess) {
    double partialSquaredNorm = getSquaredNorm(srcVector + displacementsOfPartsOfVector[currentProcess], sizesOfPartsOfVector[currentProcess]);
    MPI_Allreduce(&partialSquaredNorm, dstSquaredNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

bool isSolved(const double* partOfMatrixA, const double* vectorX, const double* vectorB, double* partOfVectorAXn, double* partOfVectorDiffAXnB, double squaredNormB, double threshold, int* sizesOfPartsOfVector, int* displacementsOfPartsOfVector, int currentProcess) {
    double squaredNormAXnB = 0.0;
    double partOfNorm = 0.0;
    getMatrixVectorMultiplication(partOfMatrixA, vectorX, partOfVectorAXn, sizesOfPartsOfVector[currentProcess]);
    getDiffOfVectors(partOfVectorAXn, vectorB + displacementsOfPartsOfVector[currentProcess], partOfVectorDiffAXnB, 1.0, sizesOfPartsOfVector[currentProcess]);
    partOfNorm = getSquaredNorm(partOfVectorDiffAXnB, sizesOfPartsOfVector[currentProcess]);
    MPI_Allreduce(&partOfNorm, &squaredNormAXnB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double resultNorm = squaredNormAXnB / squaredNormB;
    return (resultNorm < threshold * threshold) ? true : false;
}

void getNextX(double* vectorX, double* partOfVectorDiffAXnB, double* partOfVectorX1, double multiplier, int* sizesOfPartsOfVector, int* displacementsOfPartsOfVector, int currentProcess) {
    getDiffOfVectors(vectorX + displacementsOfPartsOfVector[currentProcess], partOfVectorDiffAXnB, partOfVectorX1, multiplier, sizesOfPartsOfVector[currentProcess]);
    MPI_Allgatherv(partOfVectorX1, sizesOfPartsOfVector[currentProcess], MPI_DOUBLE, vectorX, sizesOfPartsOfVector, displacementsOfPartsOfVector, MPI_DOUBLE, MPI_COMM_WORLD);
}

void fillSizesOfParts(int* sizesOfPartsOfVector, int* sizesOfPartsOfMatrix, int numberOfProcesses) {
    int fullPart = SIZE_OF_VECTOR / numberOfProcesses;
    int remainingPart = SIZE_OF_VECTOR % numberOfProcesses;

    for (int i = 0; i < numberOfProcesses; ++i) {
        sizesOfPartsOfVector[i] = fullPart;
        sizesOfPartsOfMatrix[i] = fullPart * SIZE_OF_VECTOR;
    }
    for (int i = 0; i < remainingPart; ++i) {
        sizesOfPartsOfVector[i] += 1;
        sizesOfPartsOfMatrix[i] += SIZE_OF_VECTOR;
    }
}

void fillDisplacementsOfParts(int* displacementsOfPartsOfVector, int* displacementsOfPartsOfMatrix, int* sizesOfPartsOfVector, int* sizesOfPartsOfMatrix, int numberOfProcesses) {
    for (int i = 0; i < numberOfProcesses; ++i) {
        displacementsOfPartsOfVector[i] = getShift(sizesOfPartsOfVector, i);
        displacementsOfPartsOfMatrix[i] = getShift(sizesOfPartsOfMatrix, i);
    }
}

void setDataToZero(double* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 0.0;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int numberOfProcesses, currentProcess;
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    double* vectorX = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
    double* vectorX1 = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
    double* vectorB = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
    double* vectorAX = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
    double* vectorDiffAXB = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR);
    double* matrixA = (double*)malloc(sizeof(double) * SIZE_OF_VECTOR * SIZE_OF_VECTOR);

    int* sizesOfPartsOfVector = (int*)malloc(sizeof(int) * numberOfProcesses);
    int* sizesOfPartsOfMatrix = (int*)malloc(sizeof(int) * numberOfProcesses);
    int* displacementsOfPartsOfVector = (int*)malloc(sizeof(int) * numberOfProcesses);
    int* displacementsOfPartsOfMatrix = (int*)malloc(sizeof(int) * numberOfProcesses);

    double squaredNormB = 0.0;

    double start = 0.0;
    double end = 0.0;

    if (currentProcess == ZERO_PROCESS) {
        setDataFirstVariant(matrixA, vectorB, vectorX, vectorX1, vectorAX, vectorDiffAXB);
        fillSizesOfParts(sizesOfPartsOfVector, sizesOfPartsOfMatrix, numberOfProcesses);
        fillDisplacementsOfParts(displacementsOfPartsOfVector, displacementsOfPartsOfMatrix, sizesOfPartsOfVector, sizesOfPartsOfMatrix, numberOfProcesses);

        start = MPI_Wtime();
    }

    MPI_Bcast(sizesOfPartsOfVector, numberOfProcesses, MPI_INT, ZERO_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(sizesOfPartsOfMatrix, numberOfProcesses, MPI_INT, ZERO_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(displacementsOfPartsOfVector, numberOfProcesses, MPI_INT, ZERO_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(displacementsOfPartsOfMatrix, numberOfProcesses, MPI_INT, ZERO_PROCESS, MPI_COMM_WORLD);

    double* partOfMatrixA = (double*)malloc(sizeof(double) * sizesOfPartsOfMatrix[currentProcess]);
    double* partOfVectorAX = (double*)malloc(sizeof(double) * sizesOfPartsOfVector[currentProcess]);
    double* partOfVectorDiffAXB = (double*)malloc(sizeof(double) * sizesOfPartsOfVector[currentProcess]);
    double* partOfVectorX1 = (double*)malloc(sizeof(double) * sizesOfPartsOfVector[currentProcess]);

    setDataToZero(partOfVectorAX, sizesOfPartsOfVector[currentProcess]);
    setDataToZero(partOfVectorDiffAXB, sizesOfPartsOfVector[currentProcess]);
    setDataToZero(partOfVectorX1, sizesOfPartsOfVector[currentProcess]);

    calculateSquaredNormOfVector(&squaredNormB, vectorB, sizesOfPartsOfVector, displacementsOfPartsOfVector, currentProcess);
 
    MPI_Scatterv(matrixA, sizesOfPartsOfMatrix, displacementsOfPartsOfMatrix, MPI_DOUBLE, partOfMatrixA, sizesOfPartsOfMatrix[currentProcess], MPI_DOUBLE, ZERO_PROCESS, MPI_COMM_WORLD); 
    MPI_Bcast(vectorB, SIZE_OF_VECTOR, MPI_DOUBLE, ZERO_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(vectorX, SIZE_OF_VECTOR, MPI_DOUBLE, ZERO_PROCESS, MPI_COMM_WORLD);

    while(!(isSolved(partOfMatrixA, vectorX, vectorB, partOfVectorAX, partOfVectorDiffAXB, squaredNormB, EPSILON, sizesOfPartsOfVector, displacementsOfPartsOfVector, currentProcess))) {
        getNextX(vectorX, partOfVectorDiffAXB, partOfVectorX1, TAU, sizesOfPartsOfVector, displacementsOfPartsOfVector, currentProcess);
    }
    
    if (currentProcess == ZERO_PROCESS) {
        end = MPI_Wtime();
        printf("Time spent: %.2lf seconds.\n", end - start);
        printVector(vectorX, SIZE_OF_VECTOR);
    }
    
    freeMemory(vectorX, vectorX1, vectorB, vectorAX, vectorDiffAXB, matrixA, partOfMatrixA, partOfVectorX1, partOfVectorAX, partOfVectorDiffAXB, sizesOfPartsOfVector, sizesOfPartsOfMatrix, displacementsOfPartsOfVector, displacementsOfPartsOfMatrix);
    MPI_Finalize();
    
    return 0;
}
