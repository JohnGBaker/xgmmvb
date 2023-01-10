#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;
using namespace std;

void hello(string text){
  cout<<"Hello '"<<text<<"'!"<<endl;
};

PYBIND11_MODULE(helloworld, m) {
  m.doc() = "pybind11 example plugin";
  m.def("hello", &hello, "A function which adds two numbers says hello",
	py::arg("text"));
};
