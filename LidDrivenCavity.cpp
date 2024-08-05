#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>

#define IDX(I,J) ((J)*Nx + (I))
#define IDXmin(I,J) ((J)*(n_x_loc) + (I))


#include "LidDrivenCavity.h"
#include "SolverCG.h"

LidDrivenCavity::LidDrivenCavity()
{
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

/**
*    @brief update the domain size 
*    @param xlen length of x domain
*    @param ylen length of y domain
*/
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
    UpdateDxDy();
}
void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;
}

void LidDrivenCavity::Initialise()
{

    CleanUp();
    vglob = new double[Nx*Ny]();
    sglob = new double[Nx*Ny]();
    v   = new double[n_x_loc*n_y_loc]();
    vnew = new double[n_x_loc*n_y_loc]();
    s   = new double[n_x_loc*n_y_loc]();
    cg  = new SolverCG(Nx, Ny, dx, dy,n_x_loc,n_y_loc);
    cg->parallel_val_cg(rank,size,cartgrid,coord,up_rank,down_rank,left_rank,right_rank,start_x,start_y,end_x, end_y);

    }

/**
*   @brief This function sets the parallel values from the initial solver initilisation and discretises the domain
*   @param rank rank of the processor
*   @param size the number of ranks
*   @param cartcomm communicator used for mpir
*   @param coord coordinates in cartesian topology
*   @param remx remainder x
*   @param remy remainder y
*   @param k minimum chunk x
*   @param k2 minimum chunk y
*   @param start_x local start x
*    @param start_y  local start y
*   @param end_x local end x
*   @param end_y local end y
*   @param left left neighbour rank
*   @param right right neighbour rank
*   @param up upper neighbour rank
*   @param down lower neighbour rank
*/

void LidDrivenCavity::parallel_val(int rnk, int sze,MPI_Comm comm,int tempv[2], int up,int down,int left, int right){
    this->rank = rnk;
    this->size = sze;
    this->cartgrid=comm;
    this->coord = tempv;
    this->up_rank = up;
    this->down_rank = down;
    this->left_rank = left;
    this->right_rank = right;
    this->p = pow(size,0.5);
    
    int remx = Nx%p;
    int k = (Nx)/int(p);
    int k2 = (Ny)/int(p);
    int remy = Ny%p;

    cout<<"grid spacing, coord: "<<coord[0]<<coord[1]<<endl;
    
    if (coord[1]<(remx)){
        k++;
        this->start_x = k*coord[1];
        this->end_x = k*(coord[1]+1);
    }
    else{
        this->start_x = (k+1)*remx +k*(coord[1]-remx);
        this->end_x = (k+1)*remx +k*(coord[1]-remx+1);
    }

    if (coord[0]<= (p-1)- (remy)){
        this->start_y = k2*coord[0];
        this->end_y = k2*(coord[0]+1);
    }else{
        this->start_y = (k2+1)*(coord[0]-p+remy) + coord[0]*k2; 
        this->end_y =  start_y+k2+1;
    }

    this->n_x_loc = end_x-start_x;    // local increment in x and y 
    this->n_y_loc = end_y-start_y;
}



void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T/dt);
    for (int t = 0; t < NSteps; ++t) // loop through time iterations
    {
        std::cout << "Step: " << setw(8) << t
                  << "  Time: " << setw(8) << t*dt
                  << std::endl;
  
        Advance();
    }
    ;
}

/**
*    @brief Write the solution to a file - by repiecing the global domain from the smaller domains
*    @param file string denoting the name of the file you want to write the solution to
*    @param clean_arg set to 1 to clean the dynamic memory
*/

void LidDrivenCavity::WriteSolution(std::string file,int clean_arg)
{
    double* u0 = new double[Nx*Ny]();
    double* u1 = new double[Nx*Ny]();
    double* sglob2 = new double[Nx*Ny]();   // conatiner for reduce all
    double* vglob2 = new double[Nx*Ny]();   // conatiner for reduce all

    for (int i = 0;i<n_x_loc;++i){
        for(int j =0;j<n_y_loc;++j){
            vglob[IDX((start_x+i),(start_y+j))] = vnew[IDXmin(i,j)];
            sglob[IDX((start_x+i),(start_y+j))] = s[IDXmin(i,j)];
        }
    }
    MPI_Barrier(cartgrid);
    MPI_Allreduce(vglob,vglob2,(Nx*Ny),MPI_DOUBLE,MPI_SUM,cartgrid);
    MPI_Allreduce(sglob,sglob2,(Nx*Ny),MPI_DOUBLE,MPI_SUM,cartgrid);

    if (rank == 0){
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                u0[IDX(i,j)] =  (sglob2[IDX(i,j+1)] - sglob2[IDX(i,j)]) / dy;
                u1[IDX(i,j)] = -(sglob2[IDX(i+1,j)] - sglob2[IDX(i,j)]) / dx;

            }
        }
        for (int i = 0; i < Nx; ++i) {
            u0[IDX(i,Ny-1)] = U;
        }
        

        ofstream f(file.c_str()); 
        
        cout << "Writing file " << file << std::endl;
        int k = 0;
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                k = IDX(i, j);
                f << i * dx << " " << j * dy << " " << vglob2[k] <<  " " << sglob2[k] 
                << " " << u0[k] << " " << u1[k] << std::endl;
            }
            f << std::endl;
        }
        f.close();
    }

    delete[] u0;
    delete[] u1;
    delete[] vglob2;
    delete[] sglob2;
    
    if (clean_arg == 1){
        CleanUp();
    }    
}

/**
*   @brief Print the configuration
*   @param Nx length of x domain
*   @param Ny length of y domain
*   @param Npts global grid size
*   @param dt time increment
*/
void LidDrivenCavity::PrintConfiguration()
{
    cout << "Grid size: " << Nx << " x " << Ny << endl;
    cout << "Spacing:   " << dx << " x " << dy << endl;
    cout << "Length:    " << Lx << " x " << Ly << endl;
    cout << "Grid pts:  " << Npts << endl;
    cout << "Timestep:  " << dt << endl;
    cout << "Steps:     " << ceil(T/dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}

/**
*   @brief Delete dynamic memory assigned to class instance
*   @param v vorticity
*   @param vnew time incremetned vorticity
*   @param cg gsolver cg instance
*   @param sglob global streamfunction
*/
void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] s;
        delete[] vnew;
        delete[] vglob;
        delete[] sglob;
        delete cg;
    }
}

/**
*   @brief Update spacing
*   @param dx x grid spacing 
*   @param dy y grud spacing
*   @param dxi inverse of x grid spacing
*   @param dx2i inverse of x gid spacing squared
*/
void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1); 
    dxi  = 1/dx;
    dyi  = 1/dy;
    dx2i = 1/dx/dx;
    dy2i = 1/dy/dy;
    Npts = Nx * Ny;
}


/**
* @brief calculation of vorticity and stream function at t+dt, by taking a discretised domain, sending and recieving boundary conditions between ranks while solving each local discretisation
*/

void LidDrivenCavity::Advance()
{   
    double left_s_s[n_y_loc],right_s_s[n_y_loc],up_s_s[n_x_loc],down_s_s[n_x_loc]; // sending arrays
    double left_r_s[n_y_loc],right_r_s[n_y_loc],up_r_s[n_x_loc],down_r_s[n_x_loc]; // recieving arrays

    if (left_rank != MPI_PROC_NULL){    // if neighbour exists
        for (int j = 0;j<n_y_loc;++j){
            left_s_s[j] = s[IDXmin(0,j)];
        }
        MPI_Sendrecv(left_s_s,n_y_loc,MPI_DOUBLE,left_rank,0,&left_r_s,n_y_loc,MPI_DOUBLE,left_rank,0,cartgrid,MPI_STATUS_IGNORE);
    }
    
    if (right_rank != MPI_PROC_NULL){
        for (int j = 0;j<n_y_loc;++j){
            right_s_s[j] = s[IDXmin(n_x_loc-1,j)];
        }
        MPI_Sendrecv(right_s_s,n_y_loc,MPI_DOUBLE,right_rank,0,&right_r_s,n_y_loc,MPI_DOUBLE,right_rank,0,cartgrid,MPI_STATUS_IGNORE);
    }
    
    if (up_rank != MPI_PROC_NULL){
        for (int i = 0;i<n_x_loc;++i){
            up_s_s[i] = s[IDXmin(i,n_y_loc-1)];
        }
        MPI_Sendrecv(up_s_s,n_x_loc,MPI_DOUBLE,up_rank,0,&up_r_s,n_x_loc,MPI_DOUBLE,up_rank,0,cartgrid,MPI_STATUS_IGNORE);
    }
    if (down_rank != MPI_PROC_NULL){
        for (int j = 0;j<n_x_loc;++j){
            down_s_s[j] = s[IDXmin(j,0)];
        }

        MPI_Sendrecv(down_s_s,n_x_loc,MPI_DOUBLE,down_rank,0,&down_r_s,n_x_loc,MPI_DOUBLE,down_rank,0,cartgrid,MPI_STATUS_IGNORE);

    }

    int tester = 0;   
    double str_x_p,str_x_m,str_y_p,str_y_m;     // temporary variables, for whether or not a different ranks values are needed    
    double current;
    int i,j;
     for(i =0; i<n_x_loc;++i){
        for(j = 0; j<n_y_loc;++j){ 
        
                if (down_rank == MPI_PROC_NULL && j == 0){ // lower bound, going to overwrite the left wall
                    v[IDXmin(i,j)]    = 2.0 * dy2i * (s[IDXmin(i,0)]    - s[IDXmin(i,1)]);
                    tester +=1;
                }
                if (up_rank == MPI_PROC_NULL && j == n_y_loc-1){  // upper bound - overwritten by lef
                    v[IDXmin(i,j)]  = 2.0 * dy2i * (s[IDXmin(i,n_y_loc-1)] - s[IDXmin(i,n_y_loc-2)]) - 2.0 * dyi*U;
                    tester +=1;
                }
                if (right_rank==MPI_PROC_NULL && i == n_x_loc-1){ // right bound
                    v[IDXmin(i,j)]  = 2.0 * dx2i * (s[IDXmin(n_x_loc-1,j)] - s[IDXmin(n_x_loc-2,j)]);
                    tester +=1;
                }
                if(left_rank==MPI_PROC_NULL && i==0){   // left bound
                v[IDXmin(i,j)]  = 2.0 * dx2i * (s[IDXmin(0,j)]    - s[IDXmin(1,j)]);
                tester +=1;
                }
                
                if(tester ==0){     // i.e the current grid point is not a domain boundary.
                   
                    if (i==n_x_loc-1){  // i.e need the right blocks values
                        str_x_p = right_r_s[j];     // the recieved values from right, arranged by column 
                    }else{
                        str_x_p = s[IDXmin(i+1,j)];     // i+1 stream function
                    }
                    if(i == 0){
                        str_x_m = left_r_s[j];
                    }else{
                        str_x_m = s[IDXmin(i-1,j)];      // i-1 stream function
                    }            
                    if(j==n_y_loc-1){
                        str_y_p = up_r_s[i];
                    }else{
                        str_y_p = s[IDXmin(i,j+1)];     // j+1 stream function
                    }
                    if(j==0){
                        str_y_m = down_r_s[i];
                    }else {
                        str_y_m = s[IDXmin(i,j-1)];      // j-1 stream function
                    }
            
                    current = s[IDXmin(i,j)];
                       
                    v[IDXmin(i,j)]  = dx2i*(2.0 * current - str_x_p - str_x_m)+ 1.0/dy/dy*(2.0 * current - str_y_p - str_y_m);
                       
                }
                tester = 0;
                
            }
        }
  
    double left_v_s[n_y_loc],right_v_s[n_y_loc],up_v_s[n_x_loc],down_v_s[n_x_loc];  // V - SENT 
    double left_v_r[n_y_loc],right_v_r[n_y_loc],up_v_r[n_x_loc],down_v_r[n_x_loc];  // V - RECIEVED -- i.e to use
    
    if (left_rank != MPI_PROC_NULL){
        for (int j = 0;j<n_y_loc;++j){
            left_v_s[j] = v[IDXmin(0,j)];       // container array for boundary value
        }
        MPI_Sendrecv(left_v_s,n_y_loc,MPI_DOUBLE,left_rank,0,&left_v_r,n_y_loc,MPI_DOUBLE,left_rank,0,cartgrid,MPI_STATUS_IGNORE);
    }


    if (right_rank != MPI_PROC_NULL){
        for (int j = 0;j<n_y_loc;++j){
            right_v_s[j] = v[IDXmin(n_x_loc-1,j)];
        }
        MPI_Sendrecv(right_v_s,n_y_loc,MPI_DOUBLE,right_rank,0,&right_v_r,n_y_loc,MPI_DOUBLE,right_rank,0,cartgrid,MPI_STATUS_IGNORE);
    }


    if (up_rank != MPI_PROC_NULL){
        for (int j = 0;j<n_x_loc;++j){
            up_v_s[j] = v[IDXmin(j,n_y_loc-1)];
        }
        MPI_Sendrecv(up_v_s,n_x_loc,MPI_DOUBLE,up_rank,0,&up_v_r,n_x_loc,MPI_DOUBLE,up_rank,0,cartgrid,MPI_STATUS_IGNORE);
    }


    if (down_rank != MPI_PROC_NULL){
        for (int j = 0;j<n_x_loc;++j){
            down_v_s[j] = v[IDXmin(j,0)];
        }

        MPI_Sendrecv(down_v_s,n_x_loc,MPI_DOUBLE,down_rank,0,&down_v_r,n_x_loc,MPI_DOUBLE,down_rank,0,cartgrid,MPI_STATUS_IGNORE);
    }
        



    int strty = 0;
    int strtx = 0;
    int endyj = n_y_loc;    // i.e each one should travel from their local zero to Ny_loc-1
    int endxi = n_x_loc;
    if (start_x ==0){strtx = 1;} 
    if (start_y ==0){strty = 1;}
    if (end_y == Ny){endyj -= 1;}
    if (end_x == Nx){endxi -= 1;}

   double v_x_p,v_x_m,v_y_p,v_y_m;
    #pragma omp parallel for private(str_x_p,str_x_m,str_y_p,str_y_m) schedule(dynamic) collapse(2) 
    for(int i =strtx; i<endxi;++i){
        for (int j = strty; j<endyj;++j){
            if (i==n_x_loc-1){          // i.e need the right blocks values
                v_x_p = right_v_r[j];
                str_x_p = right_r_s[j];     // the recieved values from right, arranged by column 
            }else{
                v_x_p = v[IDXmin(i+1,j)];
                str_x_p = s[IDXmin(i+1,j)];     // i+1 stream function
            }
            
            if(i == 0){
                v_x_m = left_v_r[j];
                str_x_m = left_r_s[j];
            }else{
                v_x_m = v[IDXmin(i-1,j)];
                str_x_m = s[IDXmin(i-1,j)];      // i-1 stream function
            }
            
            if(j==n_y_loc-1){
                v_y_p = up_v_r[i];
                str_y_p = up_r_s[i];
            }else{
                v_y_p = v[IDXmin(i,j+1)];
                str_y_p = s[IDXmin(i,j+1)];     // j+1 stream function
            }

            if(j==0){
                v_y_m = down_v_r[i];
                str_y_m = down_r_s[i];
            }else {
                v_y_m = v[IDXmin(i,j-1)];
                str_y_m = s[IDXmin(i,j-1)];      // j-1 stream function
            }
            
            current = v[IDXmin(i,j)];
            vnew[IDXmin(i,j)] = current + dt*(
                    ( (str_x_p- str_x_m) * 0.5 * dxi
                    *(v_y_p - v_y_m)* 0.5 * dyi)
                - ( (str_y_p - str_y_m) * 0.5 * dyi
                    *(v_x_p - v_x_m) * 0.5 * dxi)
                + nu * (v_x_p- 2.0 *current + v_x_m)*dx2i
                + nu * (v_y_p - 2.0 * current + v_y_m)*dy2i);
        }

    }

    cg->Solve(vnew,s);
    
}