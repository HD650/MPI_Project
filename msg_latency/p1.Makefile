CC=mpicc
CFLAG=-O3 -lm

p1: %.c
	$(CC) $(FLAG) -o $@ $^
