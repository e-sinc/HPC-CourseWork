#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <mpi.h>
using namespace std;

#include <cblas.h>
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include "LidDrivenCavity.h"
#include "SolverCG.h"
#define IDX(I,J) ((J)*9 + (I))


struct MPIFixture{
    public: 
        explicit MPIFixture(){
            argc = boost::unit_test::framework::master_test_suite().argc;
            argv = boost::unit_test::framework::master_test_suite().argv;
            cout<<"Initialising MPI"<<endl;
            MPI_Init(&argc,&argv);
        }
        ~MPIFixture(){
            cout<<"Finalising MPI"<<endl;
            MPI_Finalize;
        }
        int argc;
        char **argv;
};
BOOST_GLOBAL_FIXTURE(MPIFixture);


/**
* @brief sets the paralle values need for discretisation and the running of the code
*   @param left left neighbour rank
*   @param right right neighbour rank
*   @param up upper neighbour rank
*   @param down lower neighbour rank
*   @param cartcomm communicator used for mpir
*   @param rnk rank of the processor
*   @param sze the number of ranks
*   @param tempv conatainer for coordinates
*/

double Parallel_set(MPI_Comm& cartcomm,int& rnk,int& sze,int& left,int& right,int& up,int& down,int (&tempv)[2]){

    MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
    MPI_Comm_size(MPI_COMM_WORLD, &sze);
    int dims[2] = {sqrt(sze),sqrt(sze)};
    int qperiodic[2] ={0,0}; 

    
    MPI_Cart_create(MPI_COMM_WORLD, 2,dims,qperiodic, 1, &cartcomm);  // creating virtual topology pxp - with new communicator cartcomm
    MPI_Cart_coords(cartcomm,rnk,2,tempv);
    MPI_Cart_shift(cartcomm,1,1,&left,&right);
    MPI_Cart_shift(cartcomm,0,1,&down,&up);
    return 0;
}


/**
* @brief compare values from two files within tolerance
*   @param file_1 input file 1
*   @param file_2 input file 2
*   @param eps tolerance
*/

bool compare_value(const string& file_1, const string& file_2, double eps){

    ifstream read_1(file_1);
    ifstream read_2(file_2);

    double val1,val2;

    while (read_1 >> val1 && read_2 >> val2){

        if(abs(val1-val2)>eps){
            return false;
        }
    }
    return true;
}


/**
* @brief check if values in two arrays are within tolerance
*   @param arr1 array 1
*   @param arr2 array 2
*   @param tolerance tolerance
*/

bool arrays_equal(double* arr1, double* arr2, int size, double tolerance) {
    for (int i = 0; i < size; ++i) {
        if (abs(arr1[i] - arr2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}


/**
* @brief tests solvercg solve function with sinusoidal input
*/
BOOST_AUTO_TEST_CASE(Solver_test){
    double tolerance = 1e-4;
    double* v = new double[9*9];
    double* x = new double[9*9];
    double* array_file = new double[81];
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                    * sin(M_PI * k * i * 0.125)
                                    * sin(M_PI * l * j * 0.125);
        }
    }

    SolverCG* CG_test = nullptr;
    CG_test = new SolverCG(9,9,0.125,0.125,9,9);
    
    int sze,rnk,err,right,left,up,down;
    int tempv[2];
    MPI_Comm cartcomm;
    int init;

    init = Parallel_set(cartcomm,rnk,sze,left,right,up,down,tempv);
    CG_test->parallel_val_cg(rnk,sze,cartcomm,tempv,up,down,left,right,0,0,9, 9);
    CG_test->Solve(v,x);

   ifstream file("array_data.txt");
   for (int i=0; i<81;++i){
        file>>array_file[i];
   } 

    BOOST_CHECK(arrays_equal(x,array_file,81,tolerance));

    delete[] x;
    delete[] v;
    delete[] array_file;
}

/**
* @brief tests entire file and whether output is what it is supposed to be
*/
BOOST_AUTO_TEST_CASE(Wholistic){  // testing the entire file, i.e liddriven cavity and SolverCG
    
    int sze,rnk,err,right,left,up,down;
    int tempv[2];
    MPI_Comm cartcomm;

    int init;
    init = Parallel_set(cartcomm,rnk,sze,left,right,up,down,tempv);

    LidDrivenCavity* test = new LidDrivenCavity();
    test->SetDomainSize(1.0,1.0);
    test->SetGridSize(9,9);
    test->SetTimeStep(0.01);
    test->SetFinalTime(1.0);
    test->SetReynoldsNumber(10);
    test->PrintConfiguration();
    test->parallel_val(rnk,sze,cartcomm,tempv,up,down,left,right);
    test->Initialise();
    test->Integrate();
    test->WriteSolution("unit_all.txt",0);
    double tolerance = 1e-8; 

    BOOST_CHECK(compare_value("unit_all.txt","vnew_out.txt",tolerance));
    }










