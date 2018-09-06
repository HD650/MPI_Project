#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* first grid point */
#define XI 1.0
/* last grid point */
#define XF 100.0
#define LEFT 0
#define RIGHT 1

/* function declarations */
double fn(double);
void print_function_data(int, double*, double*, double*);
int main(int, char**);

int main(int argc, char* argv[]) {
    int rank;
    int numproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    int NGRID;
    if (argc > 1)
        NGRID = atoi(argv[1]);
    else {
        printf("Please specify the number of grid points.\n");
        exit(0);
    }

    int grid_i, grid_n, block_n; // starting grid point index of this node, number of grid points in this node
    block_n = ceil((double)NGRID / (double)numproc);
    block_n = block_n<1 ? 1:block_n;
    grid_i = rank * block_n;
    if (rank == numproc - 1)
    {
        grid_n = NGRID - (numproc - 1) * block_n;
        grid_n = grid_n<1 ? 1:grid_n;
    }
    else
        grid_n = block_n;

    // loop index
    int i;

    // domain array and step size
    double* xc = (double*)malloc(sizeof(double) * (block_n + 2));
    double dx = (double)(XF - XI) / (double)(NGRID - 1);

    // function array and derivative
    // the size will be dependent on the
    // number of processors used
    // to the program
    double *yc, *dyc;

    //"real" grid indices
    int imin, imax;

    imin = 1;
    imax = grid_n;

    // construct grid
    for (i = imin; i <= imax; i++)
        xc[i] = XI + (double)(grid_i + i - 1) * (double)(XF - XI) / (double)(NGRID - 1);
    // special cases for boundry values
    if (rank == 0)
        xc[0] = xc[1] - dx;
    else if (rank == numproc - 1)
        xc[grid_n + 1] = xc[grid_n] + dx;

    // allocate function arrays
    yc = (double*)malloc((block_n + 2) * sizeof(double));
    dyc = (double*)malloc((block_n + 2) * sizeof(double));

    MPI_Request send_left_request, send_right_request;

    // send boundry
    yc[imin] = fn(xc[imin]);
    if (rank > 0)
        MPI_Isend((void*)&(yc[imin]), 1, MPI_DOUBLE, rank - 1, LEFT, MPI_COMM_WORLD, &send_left_request);
    yc[imax] = fn(xc[imax]);
    if (rank < (numproc - 1))
        MPI_Isend((void*)&(yc[imax]), 1, MPI_DOUBLE, rank + 1, RIGHT, MPI_COMM_WORLD, &send_right_request);

    MPI_Request get_left_request, get_right_request;
    // get yc boundry
    if (rank == 0)
        yc[imin - 1] = fn(xc[imin - 1]);
    else
        MPI_Irecv((void*)yc, 1, MPI_DOUBLE, rank - 1, RIGHT, MPI_COMM_WORLD, &get_left_request);
    if (rank == numproc - 1)
        yc[imax + 1] = fn(xc[imax + 1]);
    else
        MPI_Irecv((void*)&(yc[imax + 1]), 1, MPI_DOUBLE, rank + 1, LEFT, MPI_COMM_WORLD, &get_right_request);

    // define the function
    for (i = imin; i <= imax; i++)
        yc[i] = fn(xc[i]);

    // compute the derivative using first-order finite differencing
    //
    //  d           f(x + h) - f(x - h)
    // ---- f(x) ~ --------------------
    //  dx                 2 * dx
    //
    for (i = imin + 1; i <= imax - 1; i++)
        dyc[i] = (yc[i + 1] - yc[i - 1]) / (2.0 * dx);

    // calculate the dyc of boundary
    MPI_Status get_left_status, get_right_status;
    if (rank != 0) {
        if (MPI_Wait(&get_left_request, &get_left_status) != MPI_SUCCESS)
            return -1;
    }
    dyc[imin] = (yc[imin + 1] - yc[imin - 1]) / (2.0 * dx);
    if (rank != (numproc - 1)) {
        if (MPI_Wait(&get_right_request, &get_right_status) != MPI_SUCCESS)
            return -1;
    }
    dyc[imax] = (yc[imax + 1] - yc[imax - 1]) / (2.0 * dx);

    if (rank == 0) {
        double *res_xc, *res_yc, *res_dyc;
        res_xc = (double*)malloc(sizeof(double) * (numproc * block_n));
        res_yc = (double*)malloc(sizeof(double) * (numproc * block_n));
        res_dyc = (double*)malloc(sizeof(double) * (numproc * block_n));

        // gather all data to root
        MPI_Gather((void*)&(dyc[imin]), block_n, MPI_DOUBLE, (void*)res_dyc, block_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather((void*)&(yc[imin]), block_n, MPI_DOUBLE, (void*)res_yc, block_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather((void*)&(xc[imin]), block_n, MPI_DOUBLE, (void*)res_xc, block_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        print_function_data(NGRID, res_xc, res_yc, res_dyc);
    } else {
        // send data to root
        MPI_Gather((void*)&(dyc[imin]), block_n, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather((void*)&(yc[imin]), block_n, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather((void*)&(xc[imin]), block_n, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // free allocated memory
    free(yc);
    free(dyc);

    MPI_Finalize();

    return 0;
}

// prints out the function and its derivative to a file
void print_function_data(int np, double* x, double* y, double* dydx) {
    int i;

    char filename[1024];
    sprintf(filename, "fn-%d.dat", np);

    FILE* fp = fopen(filename, "w");

    for (i = 0; i < np; i++) {
        fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
    }

    fclose(fp);
}
