N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0
START_GAMMA=.1

cyc_word_embeddings_cyc_sag_compile:
	g++ -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_cyc_sag \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=1 -DSTART_GAMMA=$(START_GAMMA)
cyc_word_embeddings_cyc_sag_run:
	@./word_embeddings_cyc_sag

cyc_word_embeddings_cyc_sgd_compile:
	g++ -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_cyc_sgd \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=0 -DSTART_GAMMA=$(START_GAMMA)
cyc_word_embeddings_cyc_sgd_run:
	@./word_embeddings_cyc_sgd

cyc_word_embeddings_hog:
	rm -rf word_embeddings_hog
	g++ -O3 -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread  -DHOG=1 -o word_embeddings_hog
	./word_embeddings_hog
cyc_word_embeddings_cyc:
	rm -rf word_embeddings_cyc
	g++ -O3 -std=c++11  cyclades_word_embeddings.cpp -lnuma -lpthread -DCYC=1 -o word_embeddings_cyc
	./word_embeddings_cyc