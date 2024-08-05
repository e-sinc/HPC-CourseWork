#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>

#include "SolverCG.h"
#include <mpi.h>
#include <omp.h>

#define IDX(I,J) ((J)*Nx + (I))
#define IDXmin(I,J) ((J)*n_x_loc_s + (I))


SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy, int n_x_loc, int n_y_loc)
{
    dx = pdx;
    dy = pdy;
    dx2i = pow(dx,-2);
    dy2i =pow(dy,-2);
    factor = pow(2.0*(dx2i + dy2i),-1);
    Nx = pNx;
    Ny = pNy;
    n_x_loc_s = n_x_loc;
    n_y_loc_s = n_y_loc;
    r = new double[n_x_loc_s*n_y_loc_s];
    p = new double[n_x_loc_s*n_y_loc_s];
    z = new double[n_x_loc_s*n_y_loc_s];
    t = new double[n_x_loc_s*n_y_loc_s]; //temp
}

/**
* @brief passes the same variables which were calculated into the class variables for solvercg
*   @param rank rank of the processor
*   @param size the number of ranks
*   @param coord coordinates in cartesian topology
*   @param remx remainder x
*   @param remy remainder y
*   @param k minimum chunk x
*   @param k2 minimum chunk y
*   @param start_x local start x
*    @param start_y  local start y
*   @param end_x local end x
*   @param end_y local end y
*   @param lefts left neighbour rank
*   @param right right neighbour rank
*   @param ups upper neighbour rank
*   @param downs lower neighbour rank
*/

void SolverCG::parallel_val_cg(int rank, int size,MPI_Comm comm,int tempv[2], int ups,int downs,int lefts, int rights,int startx, int starty, int endx, int endy){
    this->ranks = rank;
    this->sizes = size;
    this->cartgrids=comm;
    this->coordsg = tempv;
    this->up_ranks = ups;
    this->down_ranks = downs;
    this->left_ranks = lefts;
    this->right_ranks = rights;
    this->ps = pow(sizes,0.5);
    

    this -> start_x_s = startx;
    this -> start_y_s = starty;
    this -> end_x_s = endx;
    this -> end_y_s = endy;
    //cout<<"end y is: "<<end_y<<endl;
  

}

/**
* @brief delete dynamic memory which has been given to the class 
*/

SolverCG::~SolverCG()
{
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;
}

/**
* @brief performs conjugate gradient algorithm with aim of updating the stream function using the input vorticity, performs a set number of itereations to do so 
* @param b input array - vnew - vorticity at t+dt
* @param x input array - s - stream fuction this will be updated at the end of this funciton
*/

void SolverCG::Solve(double* b, double* x) {
    unsigned int n = n_x_loc_s*n_y_loc_s;
    int k;
    double alpha_t,alpha_d,alpha;

    double beta;
    double beta_t;
    double beta_d;
    double eps;
    double t_eps;
    double tol = 0.001;

    t_eps = cblas_dnrm2(n, b, 1); // vorticity t+dt norm - need to square, sum all and then square root again
    t_eps = t_eps*t_eps;
   
    MPI_Allreduce(&t_eps,&eps,1,MPI_DOUBLE,MPI_SUM,cartgrids);
    eps = sqrt(eps);

    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps << endl;
        return;
    }

    ApplyOperator(x, t);               // t is now vn+1 in terms of stream function
    cblas_dcopy(n, b, 1, r, 1);        // r = vn+1 from input
    ImposeBC(r);        // vn+1 is zero along all of the edges - i.e those stream functions cannot interfere with each other

    cblas_daxpy(n, -1.0, t, 1, r, 1);  // r = r-t,  b -Ax0,     vnew = b, A* stream function difference in vn+1 from streamfunction and eqn.11 (advance)
    Precondition(r, z);                 // z = r*(dx^2 + dy^2)/2
    cblas_dcopy(n, z, 1, p, 1);        // p_0 = ( vn+1,(input)- (vn+1,f(st)) )*(dx^2 + dy^2)/2 

    k = 0;
    do {
        k++;    // next iteration
        ApplyOperator(p, t);   

        
        alpha_t = cblas_ddot(n, t, 1, p, 1);  // alpha = p_k^T A p_k
        // alpha_t = 0;
        // #pragma omp parallel for reduction(+:alpha_t)
        //  for (int i = 0; i < n; ++i) {
        //      alpha_t += t[i] * p[i];
        //  }
        MPI_Allreduce(&alpha_t,&alpha_d,1,MPI_DOUBLE,MPI_SUM,cartgrids);
        //alpha_t = cblas_ddot(n, r, 1, z, 1);        
        // alpha_t = 0;
        // #pragma omp parallel for reduction(+:alpha_t)
        // for (int i = 0; i < n; ++i) {
        //      alpha_t += r[i] * z[i];
        //  }
        MPI_Allreduce(&alpha_t,&alpha,1,MPI_DOUBLE,MPI_SUM,cartgrids);
        alpha = alpha/alpha_d; // compute alpha_k

        beta_t  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k
        // beta_t = 0;
        // #pragma omp parallel for reduction(+:beta_t)
        // for (int i = 0; i < n; ++i) {
        //     beta_t += x[i] * y[i];
        // }
        MPI_Allreduce(&beta_t,&beta,1,MPI_DOUBLE,MPI_SUM,cartgrids);

        cblas_daxpy(n, alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k - streamfunction updates
        cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

        t_eps = cblas_dnrm2(n, r, 1); 
        t_eps = t_eps*t_eps;
        
        
        MPI_Allreduce(&t_eps,&eps,1,MPI_DOUBLE,MPI_SUM,cartgrids);
        eps = sqrt(eps);
        
        if (eps < tol*tol) {
            break;
        }
        Precondition(r, z);

        
        beta_t = cblas_ddot(n, r, 1, z, 1);
        // beta_t = 0;
        // #pragma omp parallel for reduction(+:beta_t)
        // for (int i = 0; i < n; ++i) {
        //     beta_t += x[i] * y[i];
        // }
        MPI_Allreduce(&beta_t,&beta_d,1,MPI_DOUBLE,MPI_SUM,cartgrids);
        beta = beta_d/beta;
        //cout<< "beta is: "<< beta<<endl;

        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, beta, p, 1, t, 1);
        cblas_dcopy(n, t, 1, p, 1);

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        cout << "FAILED TO CONVERGE" << endl;
        exit(-1);
    }

    cout << "Converged in " << k << " iterations. eps = " << eps << endl;
}


/**
* @brief solves vorticity at t+dt as a function of stream function at t and t+dt (algorithm eqn.12) 
* @param in input array
* @param out out array - is updated
*/
void SolverCG::ApplyOperator(double* in, double* out) {

    double left_s[n_y_loc_s],right_s[n_y_loc_s],up_s[n_x_loc_s],down_s[n_x_loc_s]; // sending arrays
    double left_r[n_y_loc_s],right_r[n_y_loc_s],up_r[n_x_loc_s],down_r[n_x_loc_s]; // recieving arrays

    if (left_ranks != MPI_PROC_NULL){
        for (int j = 0;j<n_y_loc_s;++j){
            left_s[j] = in[IDXmin(0,j)];
        }
        //cout<<"In solve: "<<ranks<<" my neighbour is: "<<left_ranks<<endl;
        MPI_Sendrecv(left_s,n_y_loc_s,MPI_DOUBLE,left_ranks,0,&left_r,n_y_loc_s,MPI_DOUBLE,left_ranks,0,cartgrids,MPI_STATUS_IGNORE);
    }
    
    if (right_ranks != MPI_PROC_NULL){
        for (int j = 0;j<n_y_loc_s;++j){
            right_s[j] = in[IDXmin(n_x_loc_s-1,j)];
        }
        MPI_Sendrecv(right_s,n_y_loc_s,MPI_DOUBLE,right_ranks,0,&right_r,n_y_loc_s,MPI_DOUBLE,right_ranks,0,cartgrids,MPI_STATUS_IGNORE);
    }
    
    if (up_ranks != MPI_PROC_NULL){
        for (int j = 0;j<n_x_loc_s;++j){
            up_s[j] = in[IDXmin(j,n_y_loc_s-1)];
        }
        MPI_Sendrecv(up_s,n_x_loc_s,MPI_DOUBLE,up_ranks,0,&up_r,n_x_loc_s,MPI_DOUBLE,up_ranks,0,cartgrids,MPI_STATUS_IGNORE);
    }
    
    if (down_ranks != MPI_PROC_NULL){
        for (int j = 0;j<n_x_loc_s;++j){
            down_s[j] = in[IDXmin(j,0)];
        }

        MPI_Sendrecv(down_s,n_x_loc_s,MPI_DOUBLE,down_ranks,0,&down_r,n_x_loc_s,MPI_DOUBLE,down_ranks,0,cartgrids,MPI_STATUS_IGNORE);
    }
    
        // temporary variables, for using other ranks values
    int starty = 0;
    int startx = 0;
    int endx = n_x_loc_s;         // starting blocks for generic block
    int endy = n_y_loc_s;
    double in_x_p,in_x_m,in_y_p,in_y_m;
    
  
    if (start_x_s == 0) {startx = 1;}   // special starting positions if on global boundary
    if (start_y_s == 0) {starty = 1;}
    if (end_y_s == Ny) {endy -= 1;}
    if (end_x_s == Nx) {endx -= 1;}
 

    double current;
    for (int i = startx; i < endx; ++i) {
        for (int j = starty; j < endy; ++j) {
            in_x_p = (i==n_x_loc_s-1)? right_r[j]:in[IDXmin(i+1,j)];
            in_x_m = (i == 0)? left_r[j]:in[IDXmin(i-1,j)];
            in_y_p = (j==n_y_loc_s-1)? up_r[i]:in[IDXmin(i,j+1)];
            in_y_m = (j==0)? down_r[i]:in[IDXmin(i,j-1)];
         
            current = 2.0*in[IDXmin(i,j)];
            out[IDXmin(i,j)] = ( -in_x_m + current-in_x_p)*dx2i+ (-in_y_m + current-in_y_p)*dy2i;
        }
    }
}

/**
* @brief divides all of in array by factor (class variable) except from the global boundaries - giving the out array
* @param in input array
* @param out input array
*/
void SolverCG::Precondition(double* in, double* out) {
    
    // interior
    int starty = 0;
    int startx = 0;
    int endx = n_x_loc_s;         // if wrong its because of this
    int endy = n_y_loc_s;
    double in_x_p,in_x_m,in_y_p,in_y_m;
    
  
    if (start_x_s == 0) {startx = 1;} 
    if (start_y_s == 0) {starty = 1;}
    if (end_y_s == Ny) {endy -= 1;}
    if (end_x_s == Nx) {endx -= 1;}


    for (int i=startx;i<endx;++i){
        for(int j=starty;j<endy;++j){
            out[IDXmin(i,j)] = in[IDXmin(i,j)]*factor;
        }
    }

    // boundary
    for (int i = 0;i<n_x_loc_s;++i){
        if (down_ranks == MPI_PROC_NULL){
            out[IDXmin(i,0)] = in[IDXmin(i,0)];
        }
        if(up_ranks == MPI_PROC_NULL){
            out[IDXmin(i,n_y_loc_s-1)] = in[IDXmin(i,n_y_loc_s-1)];
        }
    }
    
    for(int j=0; j<n_y_loc_s;++j){
        if (left_ranks==MPI_PROC_NULL){
            out[IDXmin(0,j)] = in[IDXmin(0,j)];
        }
        if (right_ranks==MPI_PROC_NULL){
            out[IDXmin(n_x_loc_s-1,j)] = in[IDXmin(n_x_loc_s-1,j)];
        }
    }
    

}

/**
* @brief impose the global boundary conditions to local block if applicable
* @param b input array - vnew - vorticity at t+dt
* @param x input array - s - stream fuction this will be updated at the end of this funciton
*/

void SolverCG::ImposeBC(double* inout) {
    if (down_ranks == MPI_PROC_NULL){   // global bottom wall
        for (int i = 0; i < n_x_loc_s; ++i) {
            inout[IDXmin(i, 0)] = 0.0;
        }
    }


    if (up_ranks == MPI_PROC_NULL){     // upper global wall
        for (int i = 0; i < n_x_loc_s; ++i) {
            inout[IDXmin(i, n_y_loc_s-1)] = 0.0;
        }
    }


    if(left_ranks == MPI_PROC_NULL){    // left global wall
        for (int j = 0; j < n_y_loc_s; ++j) {
            inout[IDXmin(0, j)] = 0.0;
        }
        }


    if(right_ranks == MPI_PROC_NULL){   // right global wall
        for (int j = 0; j < n_y_loc_s; ++j) {
            inout[IDXmin(n_x_loc_s - 1, j)] = 0.0;
        }
    }

    

}