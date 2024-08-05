CXX = mpicxx -fopenmp
CXXFLGS = -std=c++20 -Wall -O3 -ftree-vectorize 
HDRS = SolverCG.h LidDrivenCavity.h
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
TRGT = solver
LIBS = -llapack -lblas -lboost_program_options -lboost_unit_test_framework 
DX = Doxyfile

%.o : %.cpp $(HDRS) $(CXX)	$(CXXFLGS) -o $@ -c $<
$(TRGT) : $(OBJS)
	$(CXX)	-o $@ $^ $(LIBS)

unittests : Test.o LidDrivenCavity.o SolverCG.o
	$(CXX) $(CXXFLGS)	-o $@ $^ $(LIBS)

Test.o: Test.cpp
	$(CXX) $(CXXFLAGS) -c $<

.PHONY: clean

clean:
	-rm -f *.o $(TRGT)

doc:$(DX)
	doxygen $(DX)

$(DX): 
	doxygen -g $(DX)
	
