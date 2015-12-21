import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from subprocess import Popen, PIPE

N_REP = 1

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

def plotloss_across_epochs(epoch_range):
    cyc_avgs, hog_avgs = [], []
    cyc_epoch_time_pairs = run_cyclades_params_and_get_output("cyc_movielens_cyc", ["N_EPOCHS="+str(epoch_range)])
    hog_epoch_time_pairs = run_cyclades_params_and_get_output("cyc_movielens_hog", ["N_EPOCHS="+str(epoch_range)])

    epochs = [cyc_epoch_time_pairs[i] for i in range(0, len(cyc_epoch_time_pairs), 3)]
    cyc_losses = [cyc_epoch_time_pairs[i] for i in range(1, len(cyc_epoch_time_pairs), 3)]
    hog_losses = [hog_epoch_time_pairs[i] for i in range(1, len(hog_epoch_time_pairs), 3)]
    cyc_times = [cyc_epoch_time_pairs[i] for i in range(2, len(cyc_epoch_time_pairs), 3)]
    hog_times = [hog_epoch_time_pairs[i] for i in range(2, len(hog_epoch_time_pairs), 3)]
    
    print(epochs, cyc_times)
    #plt.plot(cyc_times, cyc_losses, color='r', label="cyc_loss_avgs")
   #plt.plot(hog_times, hog_losses, color='b', label="hog_loss_avgs")
    plt.plot(epochs, cyc_times, color='r', label="cyc")
    plt.plot(epochs, hog_times, color='b', label="hog")

    #for i, val in enumerate(epochs):
    #    plt.plot([cyc_times[i]], [cyc_losses[i]], marker="o", color="y")
    #    plt.plot([hog_times[i]], [hog_losses[i]], marker="o", color="y")

    plt.legend(loc='upper left')
    plt.savefig("figure.png")

#run_cyclades("cyc_model_dup", 10, 10, 1);
#run_cyclades("cyc", 10, 10, 1);
#run_cyclades("cyc_no_sync", 10, 10, 1);
#run_cyclades("hog", 10, 10, 1);

#plotdata(10, [1, 5, 10, 20, 50, 100])
#plotdata_across_epochs(1, list(range(1, 500, 10)))
plotloss_across_epochs(50)
