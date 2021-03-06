from __future__ import print_function
import sys
from random import shuffle

DISTANCE = 10

with open(sys.argv[1], 'r') as corpus:
    text = corpus.read()
    #text = text[100000*3:100000*20]
    #text = text[100000:]
    #text = text[:1000000]
    #text = text[:10000000]
    #text = text[:len(text)/100000]
    #text = text[:1000]
    
    words_list = list(set(text.split()))
    word_to_id = {}

    # Write word id mappings
    f = open("word_id_mappings", "w")
    for index, word in enumerate(list(set(words_list))):
        print("%d %s" % (index, word), file=f)
        word_to_id[word] = index
    f.close()

    # Construct graph
    g = {}
    words = text.strip().split(" ")
    #lines = [words[i:i+DISTANCE] for i in range(len(words))]
    #for line in lines:
    for i in range(len(words)):
        if i+DISTANCE >= len(words):
            continue
        #print("LINE: ", line)
        first_word = words[i]
        #for other_word in line:
        for k in range(DISTANCE):
            other_word = words[i+k]
            if other_word == first_word:
                continue
            a, b = tuple(sorted([first_word, other_word]))
            if (a,b) not in g:
                g[(a, b)] = 0
            g[(a, b)] += 1

    # Output graph to file
    f = open("sparse_graph", "w")
    for word_pair, occ in g.items():
        print("%d %d %d" % (word_to_id[word_pair[0]], word_to_id[word_pair[1]], occ), file=f)
    
    # Print stats
    print("N_NODES=%d" % len(set(list(words_list))))
    print("N_EDGES=%d" % len(g.items()))
        



