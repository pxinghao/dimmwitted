import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE

N_REP = 10

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


run_cyclades("cyc_model_dup", 10, 10, 1);
run_cyclades("cyc", 10, 10, 1);
run_cyclades("cyc_no_sync", 10, 10, 1);
run_cyclades("hog", 10, 10, 1);

#plotdata(10, [1, 5, 10, 20, 50, 100])
#plotdata_across_epochs(1, list(range(1, 500, 10)))
