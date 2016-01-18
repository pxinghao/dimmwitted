N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0

cyc_graph_cuts_cyc_compile:
	g++  -fopenmp -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -o graph_cuts_cyc \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=1 
cyc_graph_cuts_cyc_run:
	@./graph_cuts_cyc

cyc_graph_cuts_hog_compile:
	g++ -fopenmp -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -o graph_cuts_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAG=1 

cyc_graph_cuts_hog_run:
	@./graph_cuts_hog

cyc_graph_cuts_hog:
	rm -rf graph_cuts_hog
	g++ -fopenmp -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DHOG=1 -o graph_cuts_hog
	./graph_cuts_hog
cyc_graph_cuts_cyc:
	rm -rf graph_cuts_cyc
	g++ -fopenmp -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DCYC=1 -o graph_cuts_cyc
	./graph_cuts_cyc

cyc_graph_cuts_cyc_batch_tune:
	@for number in 2410000 2430000 2440000 2470000 2490000 ; do\
		echo Batch Size = $$number ; \
		g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DCYC=1 -DBATCH_SIZE=$$number -o graph_cuts_cyc; \
		./graph_cuts_cyc; \
	done
