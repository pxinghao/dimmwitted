N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0
START_GAMMA=.1

cyc_word_embeddings_cyc_compile:
	g++ -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_cyc \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=0 -DCYC=1 -DSAG=1 
cyc_word_embeddings_cyc_run:
	@./word_embeddings_cyc

cyc_word_embeddings_hog_compile:
	g++ -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -o word_embeddings_hog \
	-DN_EPOCHS=$(N_EPOCHS) -DBATCH_SIZE=$(BATCH_SIZE) -DNTHREAD=$(NTHREAD) -DRLENGTH=$(RLENGTH) \
	-DSHOULD_SYNC=$(SHOULD_SYNC) -DSHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=$(SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH) -DHOG=1 -DCYC=0 -DSAG=1 

cyc_word_embeddings_hog_run:
	@./word_embeddings_hog

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
	g++  -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread  -DHOG=1 -o word_embeddings_hog
	./word_embeddings_hog
cyc_word_embeddings_cyc:
	rm -rf word_embeddings_cyc
	g++ -Ofast -std=c++11  cyclades_word_embeddings.cpp -lnuma -lpthread -DCYC=1 -o word_embeddings_cyc
	./word_embeddings_cyc

cyc_word_embeddings_cyc_batch_tune:
	@for number in 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 660 680 700 720 740 760 780 800 820 840 860 880 900 920 940 960 980 1000 1020 1040 1060 1080 1100 1120 1140 1160 1180 1200 1220 1240 1260 1280 1300 1320 1340 1360 1380 1400 1420 1440 1460 1480 1500 1520 1540 1560 1580 1600 1620 1640 1660 1680 1700 1720 1740 1760 1780 1800 1820 1840 1860 1880 1900 1920 1940 1960 1980 2000 2020 2040 2060 2080 2100 2120 2140 2160 2180 2200 2220 2240 2260 2280 2300 2320 2340 2360 2380 2400 2420 2440 2460 2480 2500 2520 2540 2560 2580 2600 2620 2640 2660 2680 2700 2720 2740 2760 2780 2800 2820 2840 2860 2880 2900 2920 2940 2960 2980; do\
		echo Batch Size = $$number ; \
		g++ -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -DCYC=1 -DBATCH_SIZE=$$number -o tune_batch; \
		./tune_batch; \
	done
