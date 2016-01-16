from __future__ import print_function
import sys

DISTANCE = 10

with open(sys.argv[1], 'r') as corpus:
    text = corpus.read()
    text = text[100000:100000+100000]
    words_list = [x for x in set(text.split())]
    word_to_id = {}

    # Write word id mappings
    f = open("word_id_mappings", "w")
    for index, word in enumerate(words_list):
        print("%d %s" % (index, word), file=f)
        word_to_id[word] = index
    f.close()

    # Construct graph
    g = {}
    words = text.split()
    lines = [words[i:i+DISTANCE] for i in range(len(words))]
    for line in lines:
        first_word = line[0]
        for other_word in line:
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
    print("N_NODES=%d" % len(words_list))
    print("N_EDGES=%d" % len(g.items()))
        



