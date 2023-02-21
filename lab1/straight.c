#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ull unsigned long long
#define ZERO_PROCESS 0

int* createVector(ull length) {
    srand(time(NULL));
    int* vector = (int*)malloc(sizeof(int) * length);
    for (int i = 0; i < length; ++i) {
        vector[i] = 1 + rand() % 9;
    }
    return vector;
}

void findSum(int* vector1, int* vector2, ull* sum, ull length) {
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < length; ++j) {
            *sum += vector1[i] * vector2[j];
        }
    }
}

int main(int argc, char** argv) {
    int currentProcess = 0;
    int numberOfProcesses = 0;

    double start = 0.0;
    double end = 0.0;
    ull sum = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    
    if (!argv[1]) {
        printf("Please, input the length of vectors via first Command Line argument.\n");
        return 1;
    }
    ull length = atoi(argv[1]);
    
    int* vector1 = createVector(length);
    int* vector2 = createVector(length);

    start = MPI_Wtime();
    findSum(vector1, vector2, &sum, length);
    end = MPI_Wtime();
    if (currentProcess == ZERO_PROCESS) {
        printf("The sum is: %lld\nTime spent: %lf seconds.\n", sum, end - start);
    }

    free(vector1);
    free(vector2);

    MPI_Finalize();
    return 0;
}
