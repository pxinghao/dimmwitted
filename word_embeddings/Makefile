N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0
START_GAMMA=1e-6

cyc_word_embeddings_hog_saga_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAGA=1 -DSTART_GAMMA=$(START_GAMMA)

cyc_word_embeddings_hog_saga_run:
	@./word_embeddings_hog

cyc_word_embeddings_hog_saga_diverge_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAGA=1 -DSTART_GAMMA=$(START_GAMMA)

cyc_word_embeddings_hog_saga_diverge_run:
	@./word_embeddings_hog

cyc_word_embeddings_hog_sgd_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAGA=0 -DSTART_GAMMA=$(START_GAMMA)

cyc_word_embeddings_hog_sgd_run:
	@./word_embeddings_hog

cyc_word_embeddings_cyc_saga_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_cyc_sag \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=1  -DSTART_GAMMA=$(START_GAMMA)
cyc_word_embeddings_cyc_saga_run:
	@./word_embeddings_cyc_sag

cyc_word_embeddings_cyc_sag2_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_cyc_sag \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAGA=1 
cyc_word_embeddings_cyc_sag2_run:
	@./word_embeddings_cyc_sag

cyc_word_embeddings_cyc_sgd_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_cyc_sgd \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAGA=0 -DSTART_GAMMA=$(START_GAMMA)
cyc_word_embeddings_cyc_sgd_run:
	@./word_embeddings_cyc_sgd

cyc_word_embeddings_hog:
	rm -rf word_embeddings_hog
	g++  -Ofast -fopenmp -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread  -DHOG=1 -o word_embeddings_hog
	./word_embeddings_hog
cyc_word_embeddings_cyc:
	rm -rf word_embeddings_cyc
	g++ -Ofast -fopenmp -std=c++11  cyclades_word_embeddings.cpp -lnuma -lpthread -DCYC=1 -o word_embeddings_cyc
	./word_embeddings_cyc

cyc_word_embeddings_cyc_batch_tune:
	@for number in 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000 4250 4500 4750; do\
		echo Batch Size = $$number ; \
		g++ -fopenmp -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -DCYC=1 -DBATCH_SIZE=$$number -o tune_batch; \
		./tune_batch; \
	done

cyc_word_embeddings_serial:
	g++ -Ofast -std=c++11 cyclades_word_embeddings_serial.cpp -o serial
	./serial




# Serial

cyc_word_embeddings_serial_compile:
	g++ -Ofast -std=c++11 cyclades_word_embeddings_serial.cpp -o serial -DSHUFFLE=0
cyc_word_embeddings_serial_run:
	@./serial

cyc_word_embeddings_serial_shuffle_compile:
	g++ -Ofast -std=c++11 cyclades_word_embeddings_serial.cpp -o serial2 -DSHUFFLE=1
cyc_word_embeddings_serial_shuffle_run:
	@./serial2
