N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0

cyc_graph_cuts_hog:
	rm -rf graph_cuts_hog
	g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DHOG=1 -o graph_cuts_hog
	./graph_cuts_hog
cyc_graph_cuts_cyc:
	rm -rf graph_cuts_cyc
	g++ -Ofast -std=c++11 cyclades_graph_cuts.cpp -lnuma -lpthread -DCYC=1 -o graph_cuts_cyc
	./graph_cuts_cyc