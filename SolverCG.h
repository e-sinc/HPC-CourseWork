#pragma once
#include <mpi.h>

/**
*   @class SolverCG
*   @brief This class contains the necessary variables and functions for solving the cg
*   @param Nx the number of grid points in X-directions
*   @param dx discretisation of the x-axis (Lx/(Nx-1))
*/
class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy, int n_x_loc, int n_y_loc);
    ~SolverCG();

    void Solve(double* b, double* x);
    void parallel_val_cg(int rank, int size,MPI_Comm comm,int tempv[2], int ups,int downs,int lefts, int rights,int startx, int starty, int endx, int endy);
private:
    MPI_Comm cartgrids;
    double dx;
    double dy;
    double dx2i;
    double dy2i;
    double factor;
    int ps;
    int Nx;
    int Ny;
    int start_x_s;
    int end_x_s;
    int start_y_s;
    int end_y_s;
    int n_x_loc_s;
    int n_y_loc_s;
    int ranks;
    int sizes;
    int* coordsg = nullptr;
    int up_ranks,down_ranks,left_ranks,right_ranks;
    double* r;
    double* p;
    double* z;
    double* t;

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};