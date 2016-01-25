N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0

cyc_text_classification_hog_sag_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_text_classification.cpp -lnuma -lpthread -o text_classification_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAG=1 

cyc_text_classification_hog_sag_run:
	@./text_classification_hog

cyc_text_classification_hog_sgd_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_text_classification.cpp -lnuma -lpthread -o text_classification_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAG=0 

cyc_text_classification_hog_sgd_run:
	@./text_classification_hog

cyc_text_classification_cyc_sag_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_text_classification.cpp -lnuma -lpthread -o text_classification_cyc_sag \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=1 
cyc_text_classification_cyc_sag_run:
	@./text_classification_cyc_sag

cyc_text_classification_cyc_sag2_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_text_classification.cpp -lnuma -lpthread -o text_classification_cyc_sag \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=1 
cyc_text_classification_cyc_sag2_run:
	@./text_classification_cyc_sag

cyc_text_classification_cyc_sgd_compile:
	g++ -Ofast -fopenmp -std=c++11 cyclades_text_classification.cpp -lnuma -lpthread -o text_classification_cyc_sgd \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DK=$(K) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=0 
cyc_text_classification_cyc_sgd_run:
	@./text_classification_cyc_sgd

cyc_text_classification_hog:
	rm -rf text_classification_hog
	g++  -Ofast -fopenmp -std=c++11 cyclades_text_classification.cpp -lnuma -lpthread  -DHOG=1 -o text_classification_hog
	./text_classification_hog
cyc_text_classification_cyc:
	rm -rf text_classification_cyc
	g++ -Ofast -fopenmp -std=c++11  cyclades_text_classification.cpp -lnuma -lpthread -DCYC=1 -o text_classification_cyc
	./text_classification_cyc

cyc_text_classification_cyc_batch_tune:
	@for number in 1000 1500 2000 2500 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 8200 8400 8600 8800 9000 9200 9400 9600 9800; do\
		echo Batch Size = $$number ; \
		g++ -fopenmp -Ofast -std=c++11 cyclades_text_classification.cpp -lnuma -lpthread -DCYC=1 -DBATCH_SIZE=$$number -o tune_batch; \
		./tune_batch; \
	done

