## mpi point-to-point message latency
![latency](https://github.com/HD650/MPI_Project/raw/master/msg_latency/p1)
The plot shows that the communication latency raises linearly after the message size reaches some certain size.

It seems there is pair consistently slower than other pairs, but only occur when the message size is big.

These symptoms show that MPI will pad small package to some certain size, and some nodes are responsible for management work so it's slower than others. Also, the first communication is slower than the others, shows that MPI needs some overhead to establish a channel between nodes.

There are some very high deviations, may be the result of network fluctuation. 