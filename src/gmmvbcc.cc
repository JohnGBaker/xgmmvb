#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include "gmmvb-core.cc"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(gmmvbcc, m) {
  m.doc() = "GMMVB core components linked in C++";
  m.def("expectation_core_loop_cc", &expectation_core_loop,R"verbatim(
C++ implementation of core part of expectation step

Arguments:
 Setup: (const)
  activeComponents: vector listing which components are active
   from List
  gTol: Tolerance for ignoring insignificant points.
 Pointwise Data: (all const)
  Y: the points data
   from: numpy (Npoints,Dimension) array, const
  g: point-wise normalization sum for inactive components
   numpy (Npoints) array
 Model Data: 
  rho: component centers
    from numpy (kappa,Dimension) array
  Vnu: component shape matries (as vector<MatrixXd)) 
    from numpy (kappa,Dimension,Dimension) 
  barlamb: scalar value
  log_gamma0: point-wise component weight constant part
    from numpy (kappa) array
 Results:
  hat_gamma: normalized point-wise component weights
    from numpy (Npoints,kappa) array
)verbatim",
	py::arg("activeComponents"),
	py::arg("gTol"),
	py::arg("Y"),
	py::arg("g"),
	py::arg("rho"),
	py::arg("Vnu"),
	py::arg("barlamb"),
	py::arg("log_gamma0"),
	py::arg("hatgamma")
	);
  m.def("computeF_core_cc", &computeF_core, py::arg("hatgamma")
);
};
