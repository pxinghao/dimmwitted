import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from subprocess import Popen, PIPE
from collections import defaultdict

def tree(): return defaultdict(tree)

def run_command_with_params_and_get_output(command, args):
    # Compile first
    Popen(["make", command+"_compile"] + args, stdout=PIPE).communicate()[0]
    print(" ".join(["make", command+"_run"] + args))
    ret_val = Popen(["make", command+"_run"] + args, stdout=PIPE).communicate()[0].strip().split()
    print(ret_val)
    return ret_val

def get_values(v, command_range, epoch_range, batch_size_range, thread_range, rank_range, sync_range):
    values = []
    for c in command_range:
        for e in epoch_range:
            for b in batch_size_range:
                for t in thread_range:
                    for r in rank_range:
                        for s in sync_range:
                            values.append(v[c][e][b][t][r][s])
    return values

def draw_all_graphs(load_previous, epoch_range, batch_size_range, thread_range, rank_range, 
                    sync_range, commands, average_over_n_rep):
    total_iter = len(epoch_range) * len(batch_size_range) * len(thread_range) * len(rank_range) * len(sync_range) * len(commands) * average_over_n_rep
    average_losses = tree()
    average_gradient_times = tree()
    average_total_times = tree()

    if not load_previous:
        cur_iter = 0
        # Collect all relevant data
        for epoch in epoch_range:
            for batch_size in batch_size_range:
                for thread in thread_range:
                    for rank in rank_range:
                        for sync in sync_range:
                            for command in commands:
                                avg_overall_time, avg_gradient_time, avg_loss = 0, 0, 0
                                for i in range(average_over_n_rep):
                                    print("Iteration %d of %d" % (cur_iter, total_iter))
                                    cur_iter += 1
                                    # Run command with all params
                                    output = run_command_with_params_and_get_output(command, ["N_EPOCHS="+str(epoch), "BATCH_SIZE="+str(batch_size), "NTHREAD="+str(thread), "RLENGTH="+str(rank), "SHOULD_SYNC="+str(sync)])
                            
                                    # overall elapsed, gradient time, loss
                                    overall_time = float(output[0])
                                    gradient_time = float(output[1])
                                    loss = float(output[2])
                                
                                    avg_overall_time += overall_time
                                    avg_gradient_time += gradient_time
                                    avg_loss += loss

                                avg_overall_time /= average_over_n_rep
                                avg_gradient_time /= average_over_n_rep
                                avg_loss /= average_over_n_rep

                                average_losses[command][epoch][batch_size][thread][rank][sync] = avg_loss
                                average_gradient_times[command][epoch][batch_size][thread][rank][sync] = avg_gradient_time
                                average_total_times[command][epoch][batch_size][thread][rank][sync] = avg_overall_time
    else:
        with open('objs.pickle') as f:
            average_losses, average_gradient_times, average_total_times = pickle.load(f)

    with open('objs.pickle', 'w') as f:
        pickle.dump([average_losses, average_gradient_times, average_total_times], f)


    # Reminder: arrays of form [command][epoch][batch_size][thread][rank][sync]
    # Create overall time epoch plot
    for b in batch_size_range:
        for t in thread_range:
            for r in rank_range:
                for s in sync_range:
                    plt.figure()
                    param_desc = "batch=%d_thread=%d_rank=%d_sync=%d_avg_over=%d" % (b, t, r, s, average_over_n_rep)
                    title = "Overall_Time_Epoch_Comparison_" + param_desc
                    plt.title(title, fontsize=12)
                    plt.xlabel("Epoch")
                    plt.ylabel("Overall Time")

                    for c in commands:

                        times = get_values(average_total_times, [c], epoch_range, [b], [t], [r], [s])
                        epochs = epoch_range
                        print(epochs, times)
                        plt.plot(epochs, times, label=c)

                    plt.legend(loc="upper left")
                    plt.savefig(title + ".png")

    # Create gradient time epoch plot
    for b in batch_size_range:
        for t in thread_range:
            for r in rank_range:
                for s in sync_range:
                    plt.figure()
                    param_desc = "batch=%d_thread=%d_rank=%d_sync=%d_avg_over=%d" % (b, t, r, s, average_over_n_rep)
                    title = "Gradient_Time_Epoch_Comparison_" + param_desc
                    plt.title(title, fontsize=12)
                    plt.xlabel("Epoch")
                    plt.ylabel("Gradient Time")

                    for c in commands:

                        times = get_values(average_gradient_times, [c], epoch_range, [b], [t], [r], [s])
                        epochs = epoch_range
                        print(epochs, times)
                        plt.plot(epochs, times, label=c)

                    plt.legend(loc="upper left")
                    plt.savefig(title + ".png")


draw_all_graphs(0, [10, 50, 150], [200], [8], [500], [0, 1], ["cyc_movielens_cyc", "cyc_movielens_hog"], 5)
#draw_all_graphs(0, [10, 50, 150], [500], [8], [10, 100], [0, 1], ["cyc_movielens_cyc", "cyc_movielens_hog"], 2)
    
