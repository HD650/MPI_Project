#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MSG_SIZE 2097152+1
#define NODE_NUM 8
#define ROUND 1000

double calculateAV(double* data, int size)
{
    double sum=0;    
    for(int i=0; i<size; i++)
    {
        sum+=data[i];
    }
    double mean=sum/size;
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
    int msg_len[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152};
    MPI_Status status;	// the return status of reciever
    double* timer;

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

    // rank below NODENUM/2 are recievers, rank above NODENUM/2 are senders
    if(rank<(NODE_NUM/2))
    {
       for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
       {
            for(int ii=0; ii<ROUND; ii++)
            {
                // try to recieve a message
                MPI_Recv(message, msg_len[i], MPI_CHAR, rank+(NODE_NUM/2), 50, MPI_COMM_WORLD, &status);
            }
       }
    }
    else
    {
        timer=(double*)malloc(sizeof(double)*ROUND*(sizeof(msg_len)/sizeof(msg_len[0])));
        memset(timer, 0, sizeof(double)*ROUND*(sizeof(msg_len)/sizeof(msg_len[0])));
        struct timeval start, end;
        for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
        {
            for(int ii=0; ii<ROUND; ii++)
            {
                gettimeofday(&start, NULL);
                // send a messagek
                MPI_Send(message, msg_len[i], MPI_CHAR, rank-(NODE_NUM/2), 50, MPI_COMM_WORLD);
                gettimeofday(&end, NULL);
                timer[i*ROUND+ii]=(end.tv_sec-start.tv_sec) * 1000.0;
                timer[i*ROUND+ii]+=(end.tv_usec-start.tv_usec) / 1000.0;
            }
        }

        // calculate the result
        printf("[size]\t[std_aveage]\t[std_deviation]\n");
        for(int i=0; i<(sizeof(msg_len)/sizeof(msg_len[0])); i++)
        {
            double stddev, stdave;
            // skip first message exchange
            if(i==0)
            {
                stddev=calculateSD(timer+1+i*ROUND, ROUND-1);
                stdave=calculateAV(timer+1+i*ROUND, ROUND-1);
            }
            else
            {
                stddev=calculateSD(timer+i*ROUND, ROUND);
                stdave=calculateAV(timer+i*ROUND, ROUND);
            }
            printf("%d\t%lf\t%lf\n", msg_len[i], stdave, stddev);
        }
        free(timer);
    }
    MPI_Finalize();
}
