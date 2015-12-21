N_EPOCHS=10
GRAD_COST=1

UNAME := $(shell uname)

### LINUX ###
ifeq ($(UNAME), Linux)

ifndef CXX
CXX=g++
endif

CPP_FLAG = -O3 -std=c++11 -I./lib/libunwind-1.1/include -L./lib/numactl-2.0.9 -I./lib/numactl-2.0.9
CPP_INCLUDE = -I./src
CPP_JULIA_LIBRARY = -fPIC -lnuma -shared src/helper/julia_helper.cpp -o libdw_julia.so
CPP_LAST = -lrt -lnuma -l pthread

endif

### MAC ###
ifeq ($(UNAME), Darwin)

ifndef CXX
CXX=clang++
endif

CPP_FLAG = -O3 -std=c++11  
CPP_INCLUDE = -I./src
CPP_JULIA_LIBRARY = -dynamiclib src/helper/julia_helper.cpp -o libdw_julia.dylib
CPP_LAST = 

endif

cyc_movielens_completion:
	rm -rf movielens_completion
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -o movielens_completion
	numactl --interleave=0,1 ./movielens_completion

cyc_movielens_sgd:
	rm  -rf movielens_cyc
	g++ -Ofast -std=c++11 cyclades_movielens_sgdtest.cpp -lnuma -lpthread -o movielens_cyc
	numactl --interleave=0,1 ./movielens_cyc

cyc_model_dup_comp:
	@rm -rf cyc_model_dup
	@g++ -Ofast -std=c++11 cyclades_benchmarking.cpp -lnuma -lpthread -DN_EPOCHS=$(N_EPOCHS) -DGRAD_COST=$(GRAD_COST) -DMODEL_DUP=1 -o cyc_model_dup
cyc_no_sync_comp:
	@rm -rf cyc_no_sync
	@g++ -Ofast -std=c++11 cyclades_benchmarking.cpp -lnuma -lpthread -DN_EPOCHS=$(N_EPOCHS) -DGRAD_COST=$(GRAD_COST) -DCYC_NO_SYNC=1 -o cyc_no_sync
cyc_comp:
	@rm -rf cyc
	g++ -Ofast -std=c++11 cyclades_benchmarking.cpp -lnuma -lpthread -DN_EPOCHS=$(N_EPOCHS) -DGRAD_COST=$(GRAD_COST) -DCYCLADES=1 -o cyc
hog_comp:
	@rm -rf hog
	@g++ -Ofast -std=c++11 cyclades_benchmarking.cpp -lnuma -lpthread -DN_EPOCHS=$(N_EPOCHS) -DGRAD_COST=$(GRAD_COST) -DHOGWILD=1 -o hog

cyc_model_dup_run:
	@numactl --interleave=0,1 ./cyc_model_dup
cyc_no_sync_run:
	@numactl --interleave=0,1 ./cyc_no_sync
cyc_run:
	@numactl --interleave=0,1 ./cyc
hog_run:
	@numactl --interleave=0,1 ./hog

exp_cyc:
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) examples/example_cyclades.cpp -o example_cyclades $(CPP_LAST)

exp:
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) examples/example.cpp -o example $(CPP_LAST)

lr: lr-help.o application/dw-lr-train.cpp application/dw-lr-test.cpp
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) application/dw-lr-train.cpp -o dw-lr-train lr-help.o $(CPP_LAST)
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) application/dw-lr-test.cpp -o dw-lr-test lr-help.o $(CPP_LAST)

lr-help.o: application/dw-lr-helper.cpp 
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) application/dw-lr-helper.cpp -c -o lr-help.o $(CPP_LAST)

dep:
	cd ./lib/numactl-2.0.9; CXX=$(CXX) make; cd ../..

test_dep:

	$(CXX) -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -c ./lib/gtest-1.7.0/src/gtest_main.cc

	$(CXX) -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -c ./lib/gtest-1.7.0/src/gtest-all.cc
runtest:

	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) -I./test -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -I./examples/ -c test/glm.cc $(CPP_LAST)

	$(CXX) $(CPP_FLAG) gtest_main.o  glm.o gtest-all.o -o run_test $(CPP_LAST)

	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):./lib/numactl-2.0.9 ./run_test

julia:

	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) -I./src -I./lib/julia/src/ -I./lib/libsupport/ -I./lib/libuv/include/ -D _JULIA \
			$(CPP_JULIA_LIBRARY) $(CPP_LAST)

