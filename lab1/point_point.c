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
    int length = 0;
    int currentProcess = 0;
    int numberOfProcesses = 0;

    double start = 0.0;
    double end = 0.0;

    int* vector1;
    int* vector2;

    if (!argv[1]) {
        printf("Please, input the length of vectors via first Command Line argument.\n");
        return 1;
    }
    length = atoll(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Status status;

    if (currentProcess == ZERO_PROCESS) {
        ull partSum = 0;
        ull sum = 0;
        int partLength = length / numberOfProcesses;
        int shift = length % numberOfProcesses;
        
        vector1 = createVector(length);
        vector2 = createVector(length);

        start = MPI_Wtime();
        for (int i = 1; i < numberOfProcesses; ++i) {
            MPI_Send(vector1 + shift + partLength * i, partLength, MPI_INT, i, FIRST_VECTOR_TAG, MPI_COMM_WORLD);
            MPI_Send(vector2, length, MPI_INT, i, SECOND_VECTOR_TAG, MPI_COMM_WORLD);
        }

        findSum(vector1, vector2, &sum, shift + partLength, length);

        for (int i = 1; i < numberOfProcesses; ++i) {
            MPI_Recv(&partSum, 1, MPI_UNSIGNED_LONG_LONG, i, RESULT_TAG, MPI_COMM_WORLD, &status);
            sum += partSum;
        }
        end = MPI_Wtime();

        printf("The sum is: %lld\nTime spent: %lf seconds.\n", sum, end - start);
    }
    else {
        ull sum = 0;
        int partLength = length / numberOfProcesses;

        vector1 = (int*)malloc(sizeof(int) * partLength);
        MPI_Recv(vector1, partLength, MPI_INT, ZERO_PROCESS, FIRST_VECTOR_TAG, MPI_COMM_WORLD, &status);
        vector2 = (int*)malloc(sizeof(int) * length);
        MPI_Recv(vector2, length, MPI_INT, ZERO_PROCESS, SECOND_VECTOR_TAG, MPI_COMM_WORLD, &status);

        findSum(vector1, vector2, &sum, partLength, length);

        MPI_Send(&sum, 1, MPI_UNSIGNED_LONG_LONG, ZERO_PROCESS, RESULT_TAG, MPI_COMM_WORLD);
    }
    free(vector1);
    free(vector2);

    MPI_Finalize();
    return 0;
}
