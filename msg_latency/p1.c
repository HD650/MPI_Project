#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MSG_SIZE 1048579+1
#define NODE_NUM 8
#define ROUND 10

double calculateAV(double* data, int size)
{
    int sum=0;    
    for(int i=0; i<size; ++i)
    {
        sum+=data[i];
    }
    int mean=sum/size;
    return mean;
}

double calculateSD(double* data, int size)
{
    double sum=0.0, mean, standardDeviation=0.0;
    int i;
    for(i=0; i<size; ++i)
    {
        sum+=data[i];
    }
    mean=sum/size;
    for(i=0; i<size; ++i)
        standardDeviation+=pow(data[i]-mean,2);
    return sqrt(standardDeviation/size);
}

int main (int argc, char** argv)
{
    int rank;	// rank of this process
    int num_p;	// number of processes
    char message[MSG_SIZE];	// 2M message buffer
    int msg_len[] = {256, 1024, 4096, 16384, 65536, 262144, 1048579};
    MPI_Status status;	// the return status of reciever
    double* timer;

    // empty message, we only care about the latency
    memset(message, 0, MSG_SIZE);
    timer=(double*)malloc(sizeof(double)*ROUND*(sizeof(msg_len)/sizeof(msg_len[0])));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    if(num_p!=NODE_NUM)
    {
        printf("[master] this message latency test should run in %d nodes\n", NODE_NUM);
        return -1;
    }

    // rank below NODENUM/2 are recievers, rank above NODENUM/2 are senders
    if(rank<(NODE_NUM/2))
    {
       for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
       {
            for(int ii=0; ii<ROUND; ii++)
            {
                // try to recieve a message
                MPI_Recv(message, msg_len[i], MPI_CHAR, rank+(NODE_NUM/2), 50, MPI_COMM_WORLD, &status);
                printf("[node %d] recieved message from %d\n", rank, rank+(NODE_NUM/2));
                // send a ack signal to sender
                MPI_Send("ack", 4, MPI_CHAR, rank+(NODE_NUM/2), 50, MPI_COMM_WORLD);
            }
       }
    }
    else
    {
        struct timeval start, end;
        for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
        {
            for(int ii=0; ii<ROUND; ii++)
            {
                gettimeofday(&start, NULL);
                // send a message
                MPI_Send(message, msg_len[i], MPI_CHAR, rank-(NODE_NUM/2), 50, MPI_COMM_WORLD);
                printf("[node %d] sent size %d message to %d\n", rank, msg_len[i], rank-(NODE_NUM/2));
                // get the ack back
                MPI_Recv(message, 4, MPI_CHAR, rank-(NODE_NUM/2), 50, MPI_COMM_WORLD, &status);
                printf("[node %d] ack got\n", rank);
                gettimeofday(&end, NULL);
                timer[i*ROUND+ii]=(end.tv_sec-start.tv_sec) * 1000.0;
                timer[i*ROUND+ii]+=(end.tv_usec-start.tv_usec) / 1000.0;
            }
        }
    }
    MPI_Finalize();
    for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
    {            
        double stddev=calculateSD(timer+i*ROUND, ROUND);
        double stdave=calculateAV(timer+i*ROUND, ROUND);
        printf("[size %d]\t%lf\t%lf\n", stdave, stddev);
    }
}
