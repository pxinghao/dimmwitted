import sys

filename = "trainDmoz.txt"
FREQ_LIM = .01

f = open(filename)
vectors = []
coord_count = {}
for line in f:
    values = line.strip().split()
    category = int(values[0])
    tf = [[int(y) for y in x.split(":")] for x in values[1:]]
    for coord, freq in tf:
        if coord not in coord_count:
            coord_count[coord] = 0
        coord_count[coord] += 1
    vectors.append([category, tf])

print("ELIMINATING # OF COORDS: %d" % (len(coord_count.items())))

# Eliminate coords in more than FREQ_LIM percent docs
filtered_vectors = []
remaining_coord_count = {}
for vector in vectors:
    tf = [x for x in vector[1] if coord_count[x[0]] / float(len(vectors)) < FREQ_LIM]
    filtered_vectors.append((vector[0], tf))
    for coord, freq in tf:
        if coord not in remaining_coord_count:
            remaining_coord_count[coord] = 0
        remaining_coord_count[coord] += 1      

print("REMAINING # OF COORDS: %d" % len(remaining_coord_count))

for vector in filtered_vectors:
    category = vector[0]
    tf = vector[1]
    term_freq_pairs = [":".join([str(y) for y in x]) for x in tf]
    tf_string = " ".join(term_freq_pairs)
    print("%d %s" % (category, tf_string))

f.close()
