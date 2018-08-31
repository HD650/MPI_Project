#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define MSG_SIZE 1024*2
#define NODE_NUM 8

int main (int argc, char** argv)
{
    int rank;	// rank of this process
    int num_p;	// number of processes
    char message[MSG_SIZE];	// 2M message buffer
    int msg_len[] = {32, 64, 128, 256, 512, 1024, 2048};
    MPI_Status status;	// the return status of reciever

    // empty message, we only care about the latency
    memset(message, 0, MSG_SIZE);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    if(num_p!=NODE_NUM)
    {
        printf("[master] this message latency test should run in %d nodes\n", NODE_NUM);
        return -1;
    }

    // rank<=3: reciever, rank>3: sender
    if(rank<(NODE_NUM/2))
    {
       for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
       {
           // try to recieve a message
           MPI_Recv(message, msg_len[i], MPI_CHAR, rank+(NODE_NUM/2), 50, MPI_COMM_WORLD, &status);
           printf("[node %d] recieved message from %d\n", rank, rank+(NODE_NUM/2));
           // send a ack signal to sender
           MPI_Send("ack", 4, MPI_CHAR, rank+(NODE_NUM/2), 50, MPI_COMM_WORLD);
       } 
    }
    else
    {
       for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
       {
           // send a message
           MPI_Send(message, msg_len[i], MPI_CHAR, rank-(NODE_NUM/2), 50, MPI_COMM_WORLD);
           printf("[node %d] sent size %d message to %d\n", rank, msg_len[i], rank-(NODE_NUM/2));
           // get the ack back
           MPI_Recv(message, 4, MPI_CHAR, rank-(NODE_NUM/2), 50, MPI_COMM_WORLD, &status);
           printf("[node %d] ack got\n", rank);
       }
    }
    MPI_Finalize();
}
