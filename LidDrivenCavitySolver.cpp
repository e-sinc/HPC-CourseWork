#include <iostream>
#include <mpi.h>
#include <omp.h>

using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "LidDrivenCavity.h"

bool int_sqrt(int n){
    int root = sqrt(n);
    return root*root == n;
}

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
double Parallel_set(int argc,char** argv,MPI_Comm& cartcomm,int& rnk,int& sze,int& left,int& right,int& up,int& down,int (&tempv)[2]){
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        cout << "Failed to initialise MPI" << endl;
        return -1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
    MPI_Comm_size(MPI_COMM_WORLD, &sze);
    int dims[2] = {sqrt(sze),sqrt(sze)};
    int qperiodic[2] ={0,0}; 
    if (!int_sqrt(sze)) {
            MPI_Finalize();
            cout << "Error: Number of processes must be n^2" << endl;
        return -1;
    }
    
    MPI_Cart_create(MPI_COMM_WORLD, 2,dims,qperiodic, 1, &cartcomm);  // creating virtual topology pxp - with new communicator cartcomm
    MPI_Cart_coords(cartcomm,rnk,2,tempv);
    MPI_Cart_shift(cartcomm,1,1,&left,&right);
    MPI_Cart_shift(cartcomm,0,1,&down,&up);
    return 0;
}


int main(int argc, char **argv)
{

    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(1.0),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }

    int sze,rnk,err,right,left,up,down;
    int tempv[2];
    MPI_Comm cartcomm;

    int init;
    init = Parallel_set(argc,argv,cartcomm,rnk,sze,left,right,up,down,tempv);
    if(init == -1){
        cout<<"MPI failed to initialise"<<endl;
        return -1;
    }

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());
    solver->PrintConfiguration();
    solver->parallel_val(rnk,sze,cartcomm,tempv,up,down,left,right);

    solver->Initialise();
    solver->WriteSolution("ic.txt",0);
    
    solver->Integrate();
    
    solver->WriteSolution("final.txt",1);
    
    MPI_Finalize();
	return 0;
}