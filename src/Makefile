gmmvb: gmmvb.cc
INC=/opt/local/include

#.SUFFIXES: .c .cc .o

%.o : %.cc
	${CXX} -c ${CFLAGS} -std=c++11 $< -I${INC}

#Just a test of pybind11
helloworld:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup `python -m pybind11 --includes` pybind11helloworld.cc -o helloworld`python3-config --extension-suffix` -I/opt/local/include

gmmvbcc:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup `python -m pybind11 --includes` gmmvbcc.cc -o gmmvbcc`python3-config --extension-suffix` -I/Users/jgbaker/anaconda3/include/eigen3
