//#include <cmath>
//#include <random>
#include "Eigen/Dense"

using namespace std;

#define MatrixXd Eigen::MatrixXd
#define VectorXd Eigen::VectorXd
//The following is to align with npumpy default storage ordering
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

double computeF_core(Eigen::Ref<const RowMatrixXd> hatgamma){
  
  double glogg = 0;
  for(int j=0;j<hatgamma.cols();j++)
    for(int i=0;i<hatgamma.rows();i++)
      glogg+= hatgamma(i,j)*log(1e-300+hatgamma(i,j));
  return glogg;
};

//This just converts the the slowest part of the expectation step to C++
void expectation_core_loop(
			   const vector<int> activeComponents,
			   double gTol,
			   Eigen::Ref<const RowMatrixXd> Y,
			   Eigen::Ref<const VectorXd> g,
			   Eigen::Ref<const RowMatrixXd> rho,
			   vector<Eigen::Ref<const RowMatrixXd>> Vnu,
			   const double barlamb,
			   Eigen::Ref<const VectorXd> log_gamma0,
			   Eigen::Ref<RowMatrixXd> hatgamma){
  //Arguments:
  // Setup: (const)
  //  activeComponents: vector listing which components are active
  //   from List
  //  gTol: Tolerance for ignoring insignificant points
  // Pointwise Data: (all const)
  //  Y: the points data
  //   from: numpy (Npoints,Dimension) array, const
  //  g: point-wise normalization sum for inactive components
  //   numpy (Npoints) array
  // Model Data: 
  //  rho: component centers
  //    from numpy (kappa,Dimension) array
  //  Vnu: component shape matries (as vector<MatrixXd)) 
  //    from numpy (kappa,Dimension,Dimension) 
  //  barlamb: scalar value
  //  log_gamma0: point-wise component weight constant part
  //    from numpy (kappa) array
  // Results:
  //  hat_gamma: normalized point-wise component weights
  //    from numpy (Npoints,kappa) array
  int N=Y.rows();
  int Kactive=activeComponents.size();
  //Loop over the data for the rest of the terms and normalization
  for(int i=0;i<N;i++){
    if(g[i]<gTol)continue; //Leave insignificant hatgamma values unchanged for insignificantly overlapping data
    //First we compute log(gamma) as needed
    VectorXd log_gamma(Kactive);
    for( int k=0;k<Kactive;k++){
      int j=activeComponents[k];
      log_gamma[k]=log_gamma0[j];
      //Data-dependent terms from Eq(24) from notes
      VectorXd dy=Y.row(i)-rho.row(j);
      VectorXd vec=Vnu[j]*(dy/2.0);
      log_gamma[k] += -dy.transpose()*vec;
      //if(self.needFalt):self.gammabar[i,k]=math.exp(log_gamma[k])
    }
    //Now the exponentials of log_gamma
    VectorXd gamma(Kactive);
    gamma.setZero();
    double log_gamma_baseline=log_gamma.maxCoeff();
    for(int k=0;k<Kactive;k++){
      //Subtract off a reference baseline to avoid floating point issues
      gamma[k]=exp(log_gamma[k]-log_gamma_baseline);
    }
    //Next we normalize
    double normfac=g[i]/gamma.sum();
    for(int k=0;k<Kactive;k++){
      int j=activeComponents[k];
      hatgamma(i,j)=gamma[k]*normfac;
    }
  }
  return;
};
