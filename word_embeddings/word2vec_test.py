import sys
from math import sqrt

def cos_sim(a, b):
    top, bot1, bot2 = 0, 0, 0
    for i in range(len(a)):
        top += a[i]*b[i]
        bot1 += a[i]**2
        bot2 += b[i]**2
    return top / (sqrt(bot1) * sqrt(bot2))

trained_model_file = "trained_model"
word_to_id_mapping = "word_id_mappings"

word_to_id_map = {}
id_to_word_map = {}
f = open(word_to_id_mapping, "r")
for line in f:
    a, b = line.strip().split(" ")
    iid = int(a)
    word = str(b)
    #print(word)
    word_to_id_map[word] = iid
    id_to_word_map[iid] = word
f.close()

f = open(trained_model_file)
model = [0 for x in range(len(word_to_id_map.items()))]
for line in f:
    values = [float(x) for x in line.split()]
    model[int(values[0])] = values[1:]

def print_words_similar_to(chosen_word):
    chosen_word_id = word_to_id_map[chosen_word]
    chosen_word_vector = model[chosen_word_id]
    word_vec_pairs = [(x[0], model[x[1]]) for x in word_to_id_map.items()]
    sorted_values = sorted(word_vec_pairs, key=lambda x: -cos_sim(chosen_word_vector, x[1]))
    for i in range(10):
        print(sorted_values[i][0])

print_words_similar_to("rocket")
