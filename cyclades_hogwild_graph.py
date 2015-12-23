import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from subprocess import Popen, PIPE

N_REP = 5

def run_cyclades(command, n_rep, n_epochs, grad_cost):
    # Compile first
    Popen(["make", command+"_comp", "N_EPOCHS="+str(n_epochs), "GRAD_COST="+str(grad_cost)], stdout=PIPE).communicate()[0]

    # Take vareage of n_rep iterations
    avg = 0
    avg_loss = 0
    for i in range(n_rep):
        outputs = Popen(["make", command+"_run", "N_EPOCHS="+str(n_epochs), "GRAD_COST="+str(grad_cost)], stdout=PIPE).communicate()[0].strip().split()
        time, loss = float(outputs[0]), float(outputs[1])
        avg += time
        avg_loss += loss
    avg  = avg / float(n_rep)
    avg_loss = avg_loss / float(n_rep)

    # Return avg
    print(command+": Average Time: ", avg, " Loss ", avg_loss)
    return avg

def run_cyclades_params(command, n_rep, args):
    # Compile first
    Popen(["make", command+"_comp"] + args, stdout=PIPE).communicate()[0]

    # Take vareage of n_rep iterations
    avg = 0

    for i in range(n_rep):
        outputs = Popen(["make", command+"_run"] + args, stdout=PIPE).communicate()[0].strip().split()
        time = float(outputs[0])
        avg += time
    avg  = avg / float(n_rep)

    # Return avg
    print(command+": Average Time: ", avg)
    return avg

def run_cyclades_params_and_get_output(command, args):
    # Compile first
    Popen(["make", command+"_comp"] + args, stdout=PIPE).communicate()[0]
    print(["make", command+"_run"] + args)
    return Popen(["make", command+"_run"] + args, stdout=PIPE).communicate()[0].strip().split()

def plotdata_across_grad_cost(n_epoch, grad_cost_range):
    cyc_avgs, cyc_no_sync_avgs, hog_avgs = [], [], []
    for grad_cost in grad_cost_range:
        cyc_avg = run_cyclades("cyc", N_REP, n_epoch, grad_cost)
        cyc_no_sync_avg = run_cyclades("cyc_no_sync", N_REP, n_epoch, grad_cost)
        hog_avg = run_cyclades("hog", N_REP, n_epoch, grad_cost)
        cyc_avgs.append((grad_cost, cyc_avg))
        cyc_no_sync_avgs.append((grad_cost, cyc_no_sync_avg))
        hog_avgs.append((grad_cost, hog_avg))
    plt.plot(*zip(*cyc_avgs), color='r', label="cyc_avgs")
    plt.plot(*zip(*cyc_no_sync_avgs), color='g', label="cyc_no_sync_avgs")
    plt.plot(*zip(*hog_avgs), color='b', label="hog_avgs")
    plt.legend(loc='upper left')
    plt.savefig("figure.png")

def plotdata_across_epochs(grad_cost, epoch_range):
    cyc_avgs, cyc_no_sync_avgs, hog_avgs = [], [], []
    for n_epoch in epoch_range:
        cyc_avg = run_cyclades("cyc_movielens_cyc", N_REP, n_epoch, grad_cost)
        hog_avg = run_cyclades("cyc_movielens_hog", N_REP, n_epoch, grad_cost)
        cyc_avgs.append((n_epoch, cyc_avg))
        #cyc_no_sync_avgs.append((grad_cost, cyc_no_sync_avg))
        hog_avgs.append((n_epoch, hog_avg))
    plt.plot(*zip(*cyc_avgs), color='r', label="cyc_avgs")
    #plt.plot(*zip(*cyc_no_sync_avgs), color='g', label="cyc_no_sync_avgs")
    plt.plot(*zip(*hog_avgs), color='b', label="hog_avgs")
    plt.legend(loc='upper left')
    plt.savefig("figure.png")

def plotloss_across_epochs(epoch_range, n_rep):
    cyc_avgs, hog_avgs = [], []

    for i in range(n_rep):
        cyc_epoch_time_pairs = run_cyclades_params_and_get_output("cyc_movielens_cyc", ["N_EPOCHS="+str(epoch_range)])
        hog_epoch_time_pairs = run_cyclades_params_and_get_output("cyc_movielens_hog", ["N_EPOCHS="+str(epoch_range)])

        cyc_epoch_time_pairs = [float(x) for x in cyc_epoch_time_pairs]
        hog_epoch_time_pairs = [float(x) for x in hog_epoch_time_pairs];

        if len(cyc_avgs) == 0:
            epochs = [cyc_epoch_time_pairs[i] for i in range(0, len(cyc_epoch_time_pairs), 3)]
            cyc_losses = [cyc_epoch_time_pairs[i] for i in range(1, len(cyc_epoch_time_pairs), 3)]
            hog_losses = [hog_epoch_time_pairs[i] for i in range(1, len(hog_epoch_time_pairs), 3)]
            cyc_times = [cyc_epoch_time_pairs[i] for i in range(2, len(cyc_epoch_time_pairs), 3)]
            hog_times = [hog_epoch_time_pairs[i] for i in range(2, len(hog_epoch_time_pairs), 3)]
        else:
            cyc_losses = [cyc_losses[index] + cyc_epoch_time_pairs[i] for index, i in enumerate(range(1, len(cyc_epoch_time_pairs), 3))]
            hog_losses = [hog_losses[index] + hog_epoch_time_pairs[i] for index, i in enumerate(range(1, len(hog_epoch_time_pairs), 3))]
            cyc_times = [cyc_times[index] + cyc_epoch_time_pairs[i] for index, i in enumerate(range(2, len(cyc_epoch_time_pairs), 3))]
            hog_times = [hog_times[index] + hog_epoch_time_pairs[i] for index, i in enumerate(range(2, len(hog_epoch_time_pairs), 3))]

    print(epochs, cyc_times)

    cyc_losses = [x / float(n_rep) for x in cyc_losses]
    hog_losses = [x / float(n_rep) for x in hog_losses]
    cyc_times = [x / float(n_rep) for x in cyc_times]
    hog_times = [x / float(n_rep) for x in hog_times]
    
    plt.plot(cyc_times[:], cyc_losses[:], color='r', label="cyc_loss_avgs")
    plt.plot(hog_times[:], hog_losses[:], color='b', label="hog_loss_avgs")
    #plt.plot(epochs, cyc_times, color='r', label="cyc")
    #plt.plot(epochs, hog_times, color='b', label="hog")

    #for i, val in enumerate(epochs):
    #    plt.plot([cyc_times[i]], [cyc_losses[i]], marker="o", color="y")
    #    plt.plot([hog_times[i]], [hog_losses[i]], marker="o", color="y")

    plt.legend(loc='upper left')
    plt.savefig("figure.png")

def plotspeedups(epochs, thread_range):
    cyc_times, hog_times = [], []
    for thread in thread_range:
        cyc_out = run_cyclades_params_and_get_output("cyc_movielens_cyc", ["N_EPOCHS="+str(epochs), "NTHREAD="+str(thread)])
        hog_out = run_cyclades_params_and_get_output("cyc_movielens_hog", ["N_EPOCHS="+str(epochs), "NTHREAD="+str(thread)])
        cyc_time, cyc_loss = float(cyc_out[0]), float(cyc_out[1])
        hog_time, hog_loss = float(hog_out[0]), float(hog_out[1])
        cyc_times.append((thread, cyc_time))
        hog_times.append((thread, hog_time))
    plt.plot(*zip(*cyc_times), color="r", label="cyc_times")
    plt.plot(*zip(*hog_times), color="b", label="hog_times")
    plt.legend(loc="upper left")
    plt.savefig("figure.png")

def plot_rank(ranks, epochs, reps):
    hog_data, cyc_data = [], []
    for i, rank in enumerate(ranks):
        hog_data.append([])
        cyc_data.append([])
        for epoch in epochs:
            avg_cyc_time, avg_hog_time = 0, 0
            for rep in range(reps):
                cyc_time_loss_pairs = run_cyclades_params_and_get_output("cyc_movielens_cyc", ["N_EPOCHS="+str(epoch), "RANK="+str(rank)])
                hog_time_loss_pairs = run_cyclades_params_and_get_output("cyc_movielens_hog", ["N_EPOCHS="+str(epoch), "RANK="+str(rank)])
                cyc_time = float(cyc_time_loss_pairs[0])
                hog_time = float(hog_time_loss_pairs[0])
                avg_cyc_time += cyc_time
                avg_hog_time += hog_time
            avg_cyc_time /= float(reps)
            avg_hog_time /= float(reps)
            hog_data[i].append((epoch, hog_time))
            cyc_data[i].append((epoch, cyc_time))
    for i, rank in enumerate(ranks):
        plt.plot(*zip(*hog_data[i]), label="hog_times_rank" + str(rank))
        plt.plot(*zip(*cyc_data[i]), label="cyc_times_rank" + str(rank))
    plt.legend(loc="upper left")
    plt.savefig("figure.png")

def plot_ratio_times(ranks, threads, epochs, n_rep):
    ratio_times = []
    for j, rank in enumerate(ranks):
        ratio_times.append([])
        for i, thread in enumerate(threads):
            ratio_times[j].append([])
            for epoch in epochs:
                avg_ratio = 0
                for k in range(n_rep):
                    print("Thread: %d Epoch %d k %d" % (thread, epoch, k))
                    cyc_out = run_cyclades_params_and_get_output("cyc_movielens_cyc", ["N_EPOCHS="+str(epoch), "NTHREAD="+str(thread), "RANK="+str(rank)])
                    hog_out = run_cyclades_params_and_get_output("cyc_movielens_hog", ["N_EPOCHS="+str(epoch), "NTHREAD="+str(thread), "RANK="+str(rank)])
                    print(cyc_out, hog_out)
                    t_cyc = float(cyc_out[0])
                    t_hog = float(hog_out[0])
                    ratio = 1 / (float(t_cyc) / float(t_hog))
                    avg_ratio += ratio
                avg_ratio /= float(n_rep)
                ratio_times[j][i].append((epoch, avg_ratio))
    for j, rank in enumerate(ranks):
        for i, thread in enumerate(threads):
            plt.plot(*zip(*ratio_times[j][i]), label="hog/cyc ratio avg over " + str(n_rep) + " " + str(thread) + " threads rank" + str(rank))
    plt.legend(loc="upper left")
    plt.savefig("figure.png")

#run_cyclades("cyc_model_dup", 10, 10, 1);
#run_cyclades("cyc", 10, 10, 1);
#run_cyclades("cyc_no_sync", 10, 10, 1);
#run_cyclades("hog", 10, 10, 1);

#plotdata(10, [1, 5, 10, 20, 50, 100])

#plotloss_across_epochs(150, 1)

#plotdata_across_epochs(2, [5, 10, 50, 100, 150])
#plotspeedups(50, list(range(1, 32)));

#plot_rank([10], [5, 10, 50, 100, 150, 300, 500], N_REP)

plot_ratio_times([10, 100], [2, 4, 8], [5, 10, 50, 100, 150, 300, 500], 2)
#plot_ratio_times([10, 100], [2, 8], [5, 10], 5)
