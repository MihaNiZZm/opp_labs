#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ull unsigned long long
#define ZERO_PROCESS 0
#define FIRST_VECTOR_TAG 0
#define SECOND_VECTOR_TAG 1
#define RESULT_TAG 2

int* createVector(ull length) {
    srand(time(NULL));
    int* vector = (int*)malloc(sizeof(int) * length);
    for (int i = 0; i < length; ++i) {
        vector[i] = 1 + rand() % 9;
    }
    return vector;
}

void findSum(int* vector1, int* vector2, ull* sum, int length1, int length2) {
    for (int i = 0; i < length1; ++i) {
        for (int j = 0; j < length2; ++j) {
            *sum += vector1[i] * vector2[j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int currentProcess = 0;
    int numberOfProcesses = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    int* vector1;
    int* vector2;

    if (!argv[1]) {
        printf("Please, input the length of vectors via first Command Line argument.\n");
        return 1;
    }
    int length = atoi(argv[1]);
    int partLength = length / numberOfProcesses;
    int shift = length % numberOfProcesses;
    ull localSum = 0;
    ull sum = 0;
    
    int* partOfVector1 = (int*)malloc(sizeof(int) * partLength);
    vector2 = (int*)malloc(sizeof(int) * length);

    double start, end;

    if (currentProcess == ZERO_PROCESS) {
        vector1 = createVector(length);
        vector2 = createVector(length);

        start = MPI_Wtime();
        findSum(vector1, vector2, &localSum, shift, length);
    }
    MPI_Scatter(vector1 + shift, partLength, MPI_INT, partOfVector1, partLength, MPI_INT, ZERO_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(vector2, length, MPI_INT, ZERO_PROCESS, MPI_COMM_WORLD);

    findSum(partOfVector1, vector2, &localSum, partLength, length);

    MPI_Reduce(&localSum, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ZERO_PROCESS, MPI_COMM_WORLD);
    if (currentProcess == ZERO_PROCESS) {
        end = MPI_Wtime();

        printf("The sum is: %lld\nTime spent: %lf seconds.\n", sum, end - start);

        free(vector1);
        free(vector2);
        free(partOfVector1);
    }
    else {
        free(partOfVector1);
        free(vector2);
    }
    MPI_Finalize();
    return 0;
}
