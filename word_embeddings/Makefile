N_EPOCHS=150
BATCH_SIZE=200
NTHREAD=8
SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=0

cyc_word_embeddings_hog:
	rm -rf word_embeddings_hog
	g++ -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -DHOG=1 -o word_embeddings_hog
	./word_embeddings_hog
cyc_word_embeddings_cyc:
	rm -rf word_embeddings_cyc
	g++ -Ofast -std=c++11 cyclades_word_embeddings.cpp -lnuma -lpthread -DCYC=1 -o word_embeddings_cyc
	./word_embeddings_cyc