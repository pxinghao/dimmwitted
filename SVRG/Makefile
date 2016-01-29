N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0

cyc_matrix_inverse_hog_saga_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_matrix_inverse.cpp -lnuma   -lpthread -o matrix_inverse_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAGA=1

cyc_matrix_inverse_hog_saga_run:
	@./matrix_inverse_hog

cyc_matrix_inverse_hog_sgd_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_matrix_inverse.cpp -lnuma   -lpthread -o matrix_inverse_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAGA=0
cyc_matrix_inverse_hog_sgd_run:
	@./matrix_inverse_hog

cyc_matrix_inverse_cyc_saga_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_matrix_inverse.cpp -lnuma   -lpthread -o matrix_inverse_cyc_sag \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAGA=1
cyc_matrix_inverse_cyc_saga_run:
	@./matrix_inverse_cyc_sag

cyc_matrix_inverse_cyc_sag2_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_matrix_inverse.cpp -lnuma   -lpthread -o matrix_inverse_cyc_sag \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAGA=1
cyc_matrix_inverse_cyc_sag2_run:
	@./matrix_inverse_cyc_sag

cyc_matrix_inverse_cyc_sgd_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_matrix_inverse.cpp -lnuma   -lpthread -o matrix_inverse_cyc_sgd \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=0
cyc_matrix_inverse_cyc_sgd_run:
	@./matrix_inverse_cyc_sgd

cyc_matrix_inverse_hog:
	rm -rf matrix_inverse_hog
	g++   -Ofast -fopenmp -std=c++11 cyclades_matrix_inverse.cpp -lnuma   -lpthread  -DHOG=1 -o matrix_inverse_hog
	./matrix_inverse_hog
cyc_matrix_inverse_cyc:
	rm -rf matrix_inverse_cyc
	g++  -Ofast -fopenmp -std=c++11  cyclades_matrix_inverse.cpp -lnuma   -lpthread -DCYC=1 -o matrix_inverse_cyc
	./matrix_inverse_cyc

cyc_matrix_inverse_cyc_batch_tune:
	@for number in 1000 1500 2000 2500 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 8200 8400 8600 8800 9000 9200 9400 9600 9800; do\
		echo Batch Size = $$number ; \
		g++ -fopenmp -Ofast -std=c++11 cyclades_matrix_inverse.cpp -lnuma   -lpthread -DCYC=1 -DBATCH_SIZE=$$number -o tune_batch; \
		./tune_batch; \
	done

cyc_matrix_inverse_serial:
	rm -rf matrix_inverse_cyc_serial
	g++ -Ofast -fopenmp -std=c++11  cyclades_matrix_inverse_serial.cpp -lnuma   -lpthread -DCYC=1 -o matrix_inverse_cyc_serial
	./matrix_inverse_cyc_serial
