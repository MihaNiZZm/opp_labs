#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define L 2048
#define NUM_OF_TASKS 2048
#define NUM_OF_ITERATIONS 16
#define ASK_FOR_TASKS_TAG 128
#define SEND_NUM_OF_TASKS_TAG 256
#define SEND_TASKS_TAG 512
#define IS_DONE 8

int numOfProcs, curProcNum;
double globalResult = 0;
bool workIsDone = false;
int* taskList;
int tasksDone;
int tasksRemained;
int tasksSupplied;
pthread_mutex_t lock;

void initTasks(int* tasks, int iter) {
    for (int i = 0; i < NUM_OF_TASKS; i++){
        tasks[i] = abs(50 - i % 100) * abs(curProcNum - (iter % numOfProcs)) * L;
        // tasks[i] = abs(50 - i % 100) * abs(curProcNum) * L;
    }
}

void calculateTasks(const int* tasks) {
    int weight = 0;
    for (int i = 0; i < tasksRemained; i++) {
        pthread_mutex_lock(&lock);
        weight = tasks[i];
        pthread_mutex_unlock (&lock);
        for (int j = 0; j < weight; j++) {
            globalResult += sin(j);
        }
        pthread_mutex_lock(&lock);
        tasksDone++;
        pthread_mutex_unlock (&lock);
    }
    tasksRemained = 0;
}

void* work(void* ignored) {
    double startIter, endIter;
    double timeIter, timeIterMin, timeIterMax;
    double disbalance, disbalancePercent;

    taskList = (int*) calloc(NUM_OF_TASKS, sizeof(int));

    for (int i = 0; i < NUM_OF_ITERATIONS; i++) {

        initTasks(taskList, i);
        startIter = MPI_Wtime();
        tasksDone = 0;
        tasksSupplied = 0;
        tasksRemained = NUM_OF_TASKS;

        calculateTasks(taskList);
        int responsed = 0;
        bool enoughTasks = false;
        while (true) {
            if (enoughTasks == true) break;
            int responsedTotal = 0;
            int requested = 0;
            for (int i = 0; i < numOfProcs; i++) {

                if (i == curProcNum) continue;

                MPI_Send(&curProcNum, 1, MPI_INT, i, ASK_FOR_TASKS_TAG , MPI_COMM_WORLD);
                MPI_Recv(&responsed, 1, MPI_INT, i, SEND_NUM_OF_TASKS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                requested++;
                responsedTotal += responsed;

                if (responsedTotal == 0 && requested == numOfProcs - 1) {
                    enoughTasks = true;
                    break;
                }

                if (responsed == 0) continue;

                tasksSupplied = responsed;

                MPI_Recv(taskList, tasksSupplied, MPI_INT, i, SEND_TASKS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ;
                pthread_mutex_lock(&lock);
                tasksRemained = tasksSupplied;
                pthread_mutex_unlock(&lock);
                calculateTasks(taskList);
            }

        }

        endIter = MPI_Wtime();
        timeIter = endIter - startIter;
        MPI_Allreduce(&timeIter, &timeIterMax, 1, MPI_DOUBLE, MPI_MAX ,MPI_COMM_WORLD);
        MPI_Allreduce (&timeIter, &timeIterMin,1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        disbalance = timeIterMax - timeIterMin;
        disbalancePercent = (disbalance / timeIterMax) * 100;
        printf("\n");
        printf("curProcNum = %d\n iteration = %d\n time = %f\n taskList = %d\n globalResult = %.2f\n", curProcNum, i, timeIter, tasksDone, globalResult);
        if (curProcNum == 0) printf("iteration #%d\n disbalance = %.2f disbalancePercent = %.2f\n", i, disbalance, disbalancePercent);
        printf("\n");
    }

    pthread_mutex_lock(&lock);
    workIsDone = true;
    pthread_mutex_unlock(&lock);

    int done = IS_DONE;

    MPI_Send(&done, 1, MPI_INT, curProcNum, ASK_FOR_TASKS_TAG, MPI_COMM_WORLD) ;
    free(taskList);
    pthread_exit(NULL);
}

void receive() {
    int done;
    int tasksToSend;
    MPI_Barrier(MPI_COMM_WORLD);
    while (workIsDone == false) {
        MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, ASK_FOR_TASKS_TAG, MPI_COMM_WORLD , MPI_STATUS_IGNORE);
        if (done == IS_DONE) break;
        pthread_mutex_lock (&lock);
        int tasksLeft = tasksRemained - tasksDone;
        tasksToSend = tasksRemained / (numOfProcs * 2);
        if (tasksLeft >= 1 && tasksToSend >= 1) {
            tasksRemained -= tasksToSend;
            MPI_Send(&tasksToSend, 1, MPI_INT, done, SEND_NUM_OF_TASKS_TAG, MPI_COMM_WORLD);
            MPI_Send(&taskList[tasksRemained - tasksToSend] , tasksToSend, MPI_INT, done, SEND_TASKS_TAG, MPI_COMM_WORLD);
        } else {
            tasksToSend = 0;
            MPI_Send(&tasksToSend, 1, MPI_INT, done, SEND_NUM_OF_TASKS_TAG, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&lock);
    }
}

int main(int argc, char **argv) {
  int provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    printf("Can't set MPI_THREAD_MULTIPLE.\n");
    MPI_Finalize();
    return 1;
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &curProcNum);

  pthread_mutex_init(&lock, NULL);

  pthread_attr_t attr;

  if (pthread_attr_init(&attr) != 0) {
    printf("Can't initialize attr\n");
    MPI_Finalize();
    return 1;
  }

  if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE)) {
    printf("Can't set PTHREAD_CREATE_JOINABLE to attr.\n");
    MPI_Finalize();
    return 1;
  }

  double start, end;

  start = MPI_Wtime();

  pthread_t worker;
  if (pthread_create(&worker, &attr, work, NULL) != 0) {
    printf("Can't create work thread.\n");
    MPI_Finalize();
    return 1;
  }

  receive();

  if (pthread_join(worker, NULL) != 0) {
    printf("Can't join threads.\n");
    MPI_Finalize();
    return 1;
  }

  end = MPI_Wtime();

  if (curProcNum == 0) printf("Time spent: %.2lf seconds.\n", end - start);

  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&lock);
  MPI_Finalize();
  return 0;
}
