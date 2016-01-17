N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0

cyc_graph_cuts_cyc_compile:
	g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -o graph_cuts_cyc \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=1 
cyc_graph_cuts_cyc_run:
	@./graph_cuts_cyc

cyc_graph_cuts_hog_compile:
	g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -o graph_cuts_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAG=1 

cyc_graph_cuts_hog_run:
	@./graph_cuts_hog

cyc_graph_cuts_hog:
	rm -rf graph_cuts_hog
	g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DHOG=1 -o graph_cuts_hog
	./graph_cuts_hog
cyc_graph_cuts_cyc:
	rm -rf graph_cuts_cyc
	g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DCYC=1 -o graph_cuts_cyc
	./graph_cuts_cyc

cyc_graph_cuts_cyc_batch_tune:
	@for number in 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000; do \
		echo Batch Size = $$number ; \
		g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DCYC=1 -DBATCH_SIZE=$$number -o graph_cuts_cyc; \
		./graph_cuts_cyc; \
	done
