GRAD_COST=1 # For dataaccess only
EPOCH_LIMIT=0 # Not really used right now
TIME_LIMIT=100 # Not reall used right now

N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
RLENGTH=10
SHOULD_SYNC=0
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0
HOG=0
CYC=1

cyc_movielens_hog_regularize_compile:
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -o movielens_hog_regularize \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DMOD_REP_CYC=0 -DREGULARIZE=1
cyc_movielens_hog_regularize_run:
	@./movielens_hog_regularize

cyc_movielens_cyc_regularize_compile:
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -o movielens_cyc_regularize \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DMOD_REP_CYC=0 -DREGULARIZE=1
cyc_movielens_cyc_regularize_run:
	@./movielens_cyc_regularize

cyc_movielens_cyc_bcs_compile:
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -o movielens_cyc_bcs \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=0 -DMOD_REP_CYC=1
cyc_movielens_cyc_bcs_run:
	@./movielens_cyc_bcs

cyc_movielens_hog_compile:
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -o movielens_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0

cyc_movielens_hog_run:
	@./movielens_hog

cyc_movielens_cyc_compile:
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -o movielens_cyc \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1
cyc_movielens_cyc_run:
	@./movielens_cyc

cyc_movielens_completion_hog:
	rm -rf movielens_completion_hog
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -DHOG=1 -o movielens_completion_hog
	./movielens_completion_hog
cyc_movielens_completion_cyc:
	rm -rf movielens_completion_cyc
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -DCYC=1 -o movielens_completion_cyc
	./movielens_completion_cyc
cyc_movielens_completion_cyc_bcs:
	rm -rf movielens_completion_cyc_mod_rep
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -DMOD_REP_CYC=1 -o movielens_completion_cyc_mod_rep
	./movielens_completion_cyc_mod_rep
cyc_movielens_completion_cyc_regularize:
	rm -rf movielens_completion_cyc
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -DCYC=1 -DREGULARIZE=1 -o movielens_completion_cyc_regularize
	./movielens_completion_cyc_regularize

cyc_movielens_completion_hog_regularize:
	rm -rf movielens_completion_cyc
	g++ -Ofast -std=c++11 cyclades_movielens_completion.cpp -lnuma -lpthread -DHOG=1 -DREGULARIZE=1 -o movielens_completion_hog_regularize
	./movielens_completion_hog_regularize

