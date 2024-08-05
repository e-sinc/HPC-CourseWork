#pragma once

#include <string>
using namespace std;

class SolverCG;

/**
*    @class Lid Driven Cavity holds functions to perform the maths necessary to discretise domain, and calculate the vorticity based on stream function
*    @brief This class contains the safajkba
*/

class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);
    void parallel_val(int rank, int size,MPI_Comm comm,int tempv[2], int up,int down,int left, int right);
    void Initialise();
    void Integrate();
    void WriteSolution(string file, int clean_arg);
    void PrintConfiguration();
    

private:
    MPI_Comm cartgrid;
    double* v   = nullptr;
    double* vnew   = nullptr;
    double* s   = nullptr;
    double* sglob = nullptr;
    double* vglob = nullptr;
    int* coord = nullptr;
    int rank;
    int size;
    int start_y;
    int start_x;
    int p; 
    int end_x;
    int end_y;
    int n_x_loc,n_y_loc;
    int up_rank,down_rank,left_rank,right_rank;

    double dt   = 0.01;
    double T    = 1.0;
    double dx;
    double dy;
    double dxi;
    double dyi;
    double dx2i;
    double dy2i;
    int rdmtep;
    int    Nx   = 9;
    int    Ny   = 9;
    int    Npts = 81;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 10;
    double U    = 1.0;
    double nu   = 0.1;
    
    SolverCG* cg = nullptr;

    void CleanUp();
    void UpdateDxDy();
    void Advance();
};
