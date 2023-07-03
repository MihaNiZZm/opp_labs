#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ZERO_PROCESS 0

#define FIRST_ROW_TAG 100
#define LAST_ROW_TAG 200

#define FIELD_ROWS 256
#define FIELD_COLS 256
#define NUM_OF_ITERS 2048

void fillBreakpointFlagVector(const char* field, int tempField, int offset, char** states, int currentHistorySize, char* breakpoints) {
    for (int i = 0; i < currentHistorySize; ++i) {
        int areEqual = 1;
        for (int j = 0; areEqual && j < tempField; ++j) {
            if (field[offset + j] != states[i][offset + j]) {
                areEqual = 0;
            }
        }
        breakpoints[i] = areEqual;
    }
}

int numOfNeighbors(const char* field, int columns, int curRow, int curCol) {
    int neighbors = 0;
    for (int rowShift = -1; rowShift <= 1; ++rowShift) {
        for (int colShift = -1; colShift <= 1; ++colShift) {
            if (rowShift == colShift && rowShift == 0) {
                continue;
            }
            int pos_i = curRow + rowShift;
            int pos_j = curCol + colShift;
            if (pos_j < 0) {
                pos_j = columns - 1;
            } else if (pos_j >= columns) {
                pos_j = 0;
            }
            neighbors += field[pos_i * columns + pos_j];
        }
    }
    return neighbors;
}

void updateCell(const char* field, char* tempField, int columns, int curRow, int curCol) {
    int neighbors = numOfNeighbors(field, columns, curRow, curCol);
    tempField[curRow * columns + curCol] = field[curRow * columns + curCol];
    if (field[curRow * columns + curCol] && (neighbors < 2 || neighbors > 3)) {
        tempField[curRow * columns + curCol] = 0;
    } else if (!field[curRow * columns + curCol] && neighbors == 3) {
        tempField[curRow * columns + curCol] = 1;
    }
}

void updateMiddleRows(const char* field, char* tempField, int rows, int columns) {
    for (int i = 2; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            updateCell(field, tempField, columns, i, j);
        }
    }
}

void updateFirstRow(const char* field, char* tempField, int columns) {
    for (int i = 0; i < columns; ++i) {
        updateCell(field, tempField, columns, 1, i);
    }
}

void updateLastRow(const char* field, char* tempField, int rows, int columns) {
    for (int i = 0; i < columns; ++i) {
        updateCell(field, tempField, columns, rows, i);
    }
}

int getPrevProc(int curProc, int numOfProcs) {
    if (curProc > 0) {
        return curProc - 1;
    }
    return numOfProcs - 1;
}

int getNextProc(int curProc, int numOfProcs) {
    if (curProc < numOfProcs - 1) {
        return curProc + 1;
    }
    return 0;
}

void drawGlider(char* field, int columns, int curRow, int curCol) {
    field[curRow * columns + curCol + 1] = 1;
    field[(curRow + 2) * columns + curCol] = 1;
    field[(curRow + 2) * columns + curCol + 1] = 1;
    field[(curRow + 2) * columns + curCol + 2] = 1;
    field[(curRow + 1) * columns + curCol + 2] = 1;
}

double life(const int numOfRows, const int numOfCols, const int curProc)
{
    double start, elapsed;
    int numberOfProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    int partSize = numOfRows / numberOfProcesses * numOfCols;
    if (curProc < numOfRows % numberOfProcesses) {
        partSize += numOfCols;
    }

    char* buf = calloc(partSize + numOfCols * 2, sizeof(char));
    char* tempBuf = calloc(partSize + numOfCols * 2, sizeof(char));

    if (curProc == ZERO_PROCESS) {
        drawGlider(buf, numOfCols, 1, 0);
    }

    char* states[NUM_OF_ITERS];
    int currentHistorySize = 0;

    const int prevProc = getPrevProc(curProc, numberOfProcesses);
    const int nextProc = getNextProc(curProc, numberOfProcesses);

    char* breakpoints = calloc(NUM_OF_ITERS, sizeof(char));

    if (curProc == ZERO_PROCESS) {
        start = MPI_Wtime();
    }
    for (; currentHistorySize < NUM_OF_ITERS; ++currentHistorySize) {
        // Updating history.
        fillBreakpointFlagVector(buf, partSize, numOfCols, states, currentHistorySize, breakpoints); // 5. Calculating breakpoint-flag vector.
        states[currentHistorySize] = buf;

        // Preparing data for MPI non-blocking communications.
        MPI_Request sendToPrev, sendToNext, getFromPrev, getFromNext, vectorSharing;

        // 1. Sending the first row to the previous process.
        MPI_Isend(buf + numOfCols, numOfCols, MPI_CHAR, prevProc, FIRST_ROW_TAG, MPI_COMM_WORLD, &sendToPrev);
        
        // 2. Sending the last row to the next process.
        MPI_Isend(buf + partSize, numOfCols, MPI_CHAR, nextProc, LAST_ROW_TAG, MPI_COMM_WORLD, &sendToNext);

        // 3. Getting the last row from the previous process.
        MPI_Irecv(buf, numOfCols, MPI_CHAR, prevProc, LAST_ROW_TAG, MPI_COMM_WORLD, &getFromPrev);
        
        // 4. Getting the first row from the next process.
        MPI_Irecv(buf + partSize + numOfCols, numOfCols, MPI_CHAR, nextProc, FIRST_ROW_TAG, MPI_COMM_WORLD, &getFromNext);
        
        // 6. Reducing breakpoint-flag vector.
        MPI_Iallreduce(MPI_IN_PLACE, breakpoints, currentHistorySize, MPI_CHAR, MPI_BAND, MPI_COMM_WORLD, &vectorSharing);

        // 7. Updating middle rows (all except the first and the last ones).
        updateMiddleRows(buf, tempBuf, partSize / numOfCols, numOfCols);

        // 8. Waiting for data to be sent to the previous process.
        MPI_Wait(&sendToPrev, MPI_STATUS_IGNORE);

        // 9. Waiting for data to be received from the previous process.
        MPI_Wait(&getFromPrev, MPI_STATUS_IGNORE);

        // 10. Updating the first row.
        updateFirstRow(buf, tempBuf, numOfCols);

        // 11. Waiting for data to be sent to the next process.
        MPI_Wait(&sendToNext, MPI_STATUS_IGNORE);

        // 12. Waiting for data to be received from the next provess.
        MPI_Wait(&getFromNext, MPI_STATUS_IGNORE);

        // 13. Updating the last row.
        updateLastRow(buf, tempBuf, partSize / numOfCols, numOfCols);

        // Updating data for the next iteration.
        buf = tempBuf; // Setting new state (tempBuf) to the old one (buf).
        tempBuf = malloc(partSize + numOfCols * 2); // Allocating new memory for tempBuf (there will be no memleaks because all buffers' pointers are located in the states array).

        // 14. Waiting for vector to be shared among all of the processes.
        MPI_Wait(&vectorSharing, MPI_STATUS_IGNORE);
        
        // 15. Checking breakpoint-flag vector and ending the game if needed.
        int areEqual = 0;
        for (int i = 0; i < currentHistorySize; ++i) {
            if (breakpoints[i] == 1) {
                areEqual = 1;
                break;
            }
        }
        if (areEqual) {
            if (curProc == 0) {
                printf("Iterations: %d\n", currentHistorySize);
            }
            break;
        }
    }
    if (curProc == ZERO_PROCESS) {
        elapsed = MPI_Wtime() - start;
    }

    for (int i = 0; i < currentHistorySize; i++) {
        free(states[i]);
    }
    free(buf);
    free(tempBuf);
    free(breakpoints);

    return elapsed;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int curProc;
    MPI_Comm_rank(MPI_COMM_WORLD, &curProc);

    double elapsedTime = life(FIELD_ROWS, FIELD_COLS, curProc);
    if (curProc == ZERO_PROCESS) {
        printf("Time taken: %.2lf seconds.\n", elapsedTime);
    }

    MPI_Finalize();
    return 0;
}