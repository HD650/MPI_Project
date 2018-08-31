CC=mpicc
CFLAG=-O3

p1: %.c
	$(CC) $(FLAG) -o $@ $^
