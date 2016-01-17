import sys
import pickle
import math
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

def draw_epoch_loss_graph(should_load_from_file, epoch_range, batch_size_range, thread_range, rank_range, sync_range, commands, gammas):
    total_iter = len(batch_size_range) * len(thread_range) * len(rank_range) * len(sync_range) * len(commands)
    cur_iter = 0
    loss_values = tree()
    overall_time_values = tree()
    gradient_time_values = tree()

    if not should_load_from_file:
        for b in batch_size_range:
            for t in thread_range:
                for r in rank_range:
                    for s in sync_range:
                        for ii, c in enumerate(commands):
                            print("Iteration %d of %d" % (cur_iter, total_iter))
                            cur_iter += 1
                            output = run_command_with_params_and_get_output(c, ["N_EPOCHS="+str(epoch_range), "BATCH_SIZE="+str(b), "NTHREAD="+str(t), "RLENGTH="+str(r), "SHOULD_SYNC="+\
                                                                                    str(s), "SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=1", "START_GAMMA="+str(gammas[ii])])
                            values = [float(x) for x in output]
                            losses = [values[i] for i in range(0, len(values), 3)]
                            overall_times = [values[i] for i in range(1, len(values), 3)]
                            gradient_times = [values[i] for i in range(2, len(values), 3)]
                            loss_values[c][epoch_range][b][t][r][s] = losses
                            overall_time_values[c][epoch_range][b][t][r][s] = overall_times
                            gradient_time_values[c][epoch_range][b][t][r][s] = gradient_times
    else:
        with open('objs3.pickle') as f:
            loss_values, overall_time_values, gradient_time_values = pickle.load(f)
    with open('objs3.pickle', "w") as f:
        pickle.dump([loss_values, overall_time_values, gradient_time_values], f)

    for b in batch_size_range:
        for t in thread_range:
            for r in rank_range:
                title = "Epoch_Loss_batch=%d_thread=%d_rank=%d" % (b, t, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                for s in sync_range:
                    for c in commands:
                        losses = loss_values[c][epoch_range][b][t][r][s]
                        epochs = list(range(1, epoch_range+1))
                        
                        plt.plot(epochs, losses, label=c+" sync="+str(s), marker='o')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend(loc="upper right", fontsize=8)
                plt.savefig(title + ".png")
                plt.clf()    

def draw_time_loss_graph(should_load_from_file, epoch_range, batch_size_range, thread_range, rank_range, sync_range, commands):
    total_iter = len(batch_size_range) * len(thread_range) * len(rank_range) * len(sync_range) * len(commands)
    cur_iter = 0
    loss_values = tree()
    overall_time_values = tree()
    gradient_time_values = tree()

    if not should_load_from_file:
        for b in batch_size_range:
            for t in thread_range:
                for r in rank_range:
                    for s in sync_range:
                        for c in commands:
                            print("Iteration %d of %d" % (cur_iter, total_iter))
                            cur_iter += 1
                            output = run_command_with_params_and_get_output(c, ["N_EPOCHS="+str(epoch_range), "BATCH_SIZE="+str(b), "NTHREAD="+str(t), "RLENGTH="+str(r), "SHOULD_SYNC="+\
                                                                                    str(s), "SHOULD_PRINT_LOSS_TIME_EVERY_EPOCH=1"])
                            values = [float(x) for x in output]
                            losses = [values[i] for i in range(0, len(values), 3)]
                            overall_times = [values[i] for i in range(1, len(values), 3)]
                            gradient_times = [values[i] for i in range(2, len(values), 3)]
                            loss_values[c][epoch_range][b][t][r][s] = losses
                            overall_time_values[c][epoch_range][b][t][r][s] = overall_times
                            gradient_time_values[c][epoch_range][b][t][r][s] = gradient_times
    else:
        with open('objs2.pickle') as f:
            loss_values, overall_time_values, gradient_time_values = pickle.load(f)
    with open('objs2.pickle', "w") as f:
        pickle.dump([loss_values, overall_time_values, gradient_time_values], f)

    for b in batch_size_range:
        for t in thread_range:
            for r in rank_range:
                title = "Overall_Time_Loss_batch=%d_thread=%d_rank=%d" % (b, t, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.xlabel("Time")
                plt.ylabel("Loss")
                for s in sync_range:
                    for c in commands:
                        times = overall_time_values[c][epoch_range][b][t][r][s]
                        losses = loss_values[c][epoch_range][b][t][r][s]
                        if 'hog' in c:
                            if s:
                                plt.plot(times, losses, label=c)
                        else:
                            plt.plot(times, losses, label=c+" sync="+str(s))
                plt.yscale('log')
                #plt.xscale('log')
                plt.legend(loc="upper right", fontsize=8)
                plt.savefig(title + ".png")
                plt.clf()

    for b in batch_size_range:
        for t in thread_range:
            for r in rank_range:
                title = "Gradient_Time_Loss_batch=%d_thread=%d_rank=%d" % (b, t, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.xlabel("Time")
                plt.ylabel("Loss")
                for s in sync_range:
                    for c in commands:
                        times = gradient_time_values[c][epoch_range][b][t][r][s]
                        losses = loss_values[c][epoch_range][b][t][r][s]
                        if 'hog' in c:
                            if s:
                                plt.plot(times, losses, label=c)
                        else:
                            plt.plot(times, losses, label=c+" sync="+str(s))
                plt.yscale('log')
                #plt.xscale('log')
                plt.legend(loc="upper right", fontsize=8)
                plt.savefig(title + ".png")
                plt.clf()

    
    hog_command = [x for x in commands if 'hog' in x]
    if len(hog_command) != 0:
        hog_command = hog_command[0]
    else:
        return
    for b in batch_size_range:
        for t in thread_range:
            for r in rank_range:
                title = "Gradient_Time_Loss_Ratios_batch=%d_thread=%d_rank=%d" % (b, t, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.xlabel("Time")
                plt.ylabel("Loss(hog)/Loss(cyc)")
                for s in sync_range:
                    for c in commands:
                        if c == hog_command:
                            continue
                        hog_times = gradient_time_values[hog_command][epoch_range][b][t][r][s]
                        cyc_times = gradient_time_values[c][epoch_range][b][t][r][s]
                        hog_losses = loss_values[hog_command][epoch_range][b][t][r][s]
                        cyc_losses = loss_values[c][epoch_range][b][t][r][s]
                        # Compute cyc losses -- the best loss cyc achieved by hog's time
                        cyc_losses_aligned = []
                        for i1, t1 in enumerate(hog_times):
                            best_loss = 1000000000
                            for i2, t2 in enumerate(cyc_times):
                                if t2 > t1:
                                    break
                                best_loss = min(best_loss, cyc_losses[i2])
                            cyc_losses_aligned.append(best_loss)
                        loss_ratio = [hog_losses[i] / cyc_losses_aligned[i] for i in range(len(hog_losses))]
                        plt.plot(hog_times, loss_ratio, label=c+" sync="+str(s))
                plt.legend(loc="upper right", fontsize=8)
                plt.savefig(title + ".png")
                plt.clf()

    for b in batch_size_range:
        for t in thread_range:
            for r in rank_range:
                title = "Overall_Time_Loss_Ratios_batch=%d_thread=%d_rank=%d" % (b, t, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.xlabel("Time")
                plt.ylabel("Loss(hog)/Loss(cyc)")
                for s in sync_range:
                    for c in commands:
                        if hog_command == c:
                            continue
                        hog_times = overall_time_values[hog_command][epoch_range][b][t][r][s]
                        cyc_times = overall_time_values[c][epoch_range][b][t][r][s]
                        hog_losses = loss_values[hog_command][epoch_range][b][t][r][s]
                        cyc_losses = loss_values[c][epoch_range][b][t][r][s]
                        # Compute cyc losses -- the best loss cyc achieved by hog's time
                        cyc_losses_aligned = []
                        for i1, t1 in enumerate(hog_times):
                            best_loss = 1000000000
                            for i2, t2 in enumerate(cyc_times):
                                if t2 > t1:
                                    break
                                best_loss = min(best_loss, cyc_losses[i2])
                            cyc_losses_aligned.append(best_loss)
                        loss_ratio = [hog_losses[i] / cyc_losses_aligned[i] for i in range(len(hog_losses))]
                        plt.plot(hog_times, loss_ratio, label=c+" sync="+str(s))
                plt.legend(loc="upper right", fontsize=8)
                plt.savefig(title + ".png")
                plt.clf()

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


    """# Reminder: arrays of form [command][epoch][batch_size][thread][rank][sync]
                    plt.clf()"""
    for (time_data, label) in [(average_gradient_times, "Gradient Time"), (average_total_times, "Overall Time")]:
        for r in rank_range:
            for b in batch_size_range:
                f, plots = plt.subplots(1, len(thread_range), sharex=True, sharey=True)
                title = "Epoch_%s_Plot_Batch=%d_Rank=%d" % (label, b, r)
                f.suptitle(title, fontsize=12) 
                for index, t in enumerate(thread_range):
                    plots[index].set_title("%d threads" % t)
                    for s in sync_range:
                        for c in commands:
                            plots[index].set_xlabel("Epoch")
                            plots[index].tick_params(axis='both', which='major', labelsize=5)
                            plots[index].tick_params(axis='both', which='minor', labelsize=5)
                            plots[index].set_ylabel(label)
                            times = get_values(time_data, [c], epoch_range, [b], [t], [r], [s])
                            epochs = epoch_range
                            low = min(times)
                            high = max(times)
                            if 'hog' in c:
                                if s == 0:
                                    plots[index].plot(epochs, times, label=c)
                            else:
                                plots[index].plot(epochs, times, label=c+" sync="+str(s))
                        #plots[index].set_ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
                    plots[index].legend(loc="upper left", fontsize=5)
            #f.subplots_adjust(hspace=0)
                f.tight_layout()
                f.subplots_adjust(top=.85)
                f.savefig(title+".png")
                f.clf()
    for (time_data, label) in [(average_gradient_times, "Gradient Time"), (average_total_times, "Overall Time")]:
        for r in rank_range:
            for b in batch_size_range:
                f, plots = plt.subplots(1, len(epoch_range), sharex=True, sharey=True)
                title = "Thread_%s_Plot_Batch=%d_Rank=%d" % (label, b, r)
                f.suptitle(title, fontsize=12) 
                for index, e in enumerate(epoch_range):
                    plots[index].set_title("%d epoch" % e)
                    for s in sync_range:
                        for c in commands:
                            plots[index].tick_params(axis='both', which='major', labelsize=8)
                            plots[index].tick_params(axis='both', which='minor', labelsize=8)
                            plots[index].set_xlabel("Thread")
                            plots[index].set_ylabel(label)
                            times = get_values(time_data, [c], [e], [b], thread_range, [r], [s])
                            threads = thread_range
                            low = min(times)
                            high = max(times)
                            if 'hog' in c:
                                if s == 0:
                                    plots[index].plot(threads, times, label=c)
                            else:
                                plots[index].plot(threads, times, label=c+" sync="+str(s))
                        #plots[index].set_ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
                    plots[index].legend(loc="upper left", fontsize=5)
            #f.subplots_adjust(hspace=0)
                f.tight_layout()
                f.subplots_adjust(top=.85)
                f.savefig(title+".png")
                f.clf()

    ########################################
    # TIME RATIOS OVER 1 THREAD
    ########################################
    if 1 in thread_range:
        for r in rank_range:
            for b in batch_size_range:
                for e in epoch_range:
                    for s in sync_range:
                        for c in commands:
                            if 'hog' in c and not s: 
                                continue
                            title = ""
                            if 'hog' in c:
                                title = "Overall_Speedup_Over_Serial_%s_Batch=%d_Epoch=%d_Rank=%d" % (c, b, e, r)
                            else:
                                title = "Overall_Speedup_Over_Serial_%s_Sync=%d_Batch=%d_Epoch=%d_Rank=%d" % (c, s, b, e, r)
                            plt.figure()
                            plt.title(title, fontsize=12)
                            plt.ylabel("Serial_Time/Time_With_N_Threads")
                            plt.xlabel("N")
                            base_time = average_total_times[c][e][b][1][r][s]
                            time_values = get_values(average_total_times, [c], [e], [b], thread_range, [r], [s])
                            time_ratios = [float(base_time)/x for x in time_values]
                            plt.plot(thread_range, time_ratios)
                            plt.legend(loc="upper left", fontsize=5)
                            plt.savefig(title+".png")
                            plt.clf()
        for r in rank_range:
            for b in batch_size_range:
                for e in epoch_range:
                    for s in sync_range:
                        for c in commands:
                            if 'hog' in c and not s: 
                                continue
                            title = ""
                            if 'hog' in c:
                                title = "Gradient_Speedup_Over_Serial_%s_Batch=%d_Epoch=%d_Rank=%d" % (c, b, e, r)
                            else:
                                title = "Gradient_Speedup_Over_Serial_%s_Sync=%d_Batch=%d_Epoch=%d_Rank=%d" % (c, s, b, e, r)
                            plt.figure()
                            plt.title(title, fontsize=12)
                            plt.ylabel("Serial_Time/Time_With_N_Threads")
                            plt.xlabel("N")
                            base_time = average_gradient_times[c][e][b][1][r][s]
                            time_values = get_values(average_total_times, [c], [e], [b], thread_range, [r], [s])
                            time_ratios = [float(base_time)/x for x in time_values]
                            plt.plot(thread_range, time_ratios)
                            plt.legend(loc="upper left", fontsize=5)
                            plt.savefig(title+".png")
                            plt.clf()

    ########################################
    # TIME RATIOS OVER HOG PER EPOCH
    ########################################
    hog_command = [x for x in commands if 'hog' in x][0]
    for t in thread_range:
        for r in rank_range:
            for b in batch_size_range:
                title = "Overall_Time_Ratios_Over_Hog_Per_Epoch_Batch=%d_Thread=%d_Rank=%d" % (b, t, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.ylabel("Hog Time / Cyclades Time")
                plt.xlabel("Epoch")
                for s in sync_range:
                    for c in commands:
                        if 'hog' not in c:
                            baseline_hog_times = get_values(average_total_times, [hog_command], epoch_range, [b], [t], [r], [s])
                            times = get_values(average_total_times, [c], epoch_range, [b], [t], [r], [s])
                            ratio_times = [float(baseline_hog_times[i]) / float(times[i]) for i in range(len(times))]
                            plt.plot(epoch_range, ratio_times, label=c+"_sync="+str(s))
                plt.legend(loc="upper left", fontsize=5)
                plt.savefig(title+".png")
                plt.clf()

    hog_command = [x for x in commands if 'hog' in x][0]
    for t in thread_range:
        for r in rank_range:
            for b in batch_size_range:
                title = "Gradient_Time_Ratios_Over_Hog_Per_Epoch_Batch=%d_Thread=%d_Rank=%d" % (b, t, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.ylabel("Hog Time / Cyclades Time")
                plt.xlabel("Epoch")
                for s in sync_range:
                    for c in commands:
                        if 'hog' not in c:
                            baseline_hog_times = get_values(average_total_times, [hog_command], epoch_range, [b], [t], [r], [s])
                            times = get_values(average_gradient_times, [c], epoch_range, [b], [t], [r], [s])
                            ratio_times = [float(baseline_hog_times[i]) / float(times[i]) for i in range(len(times))]
                            plt.plot(epoch_range, ratio_times, label=c+"_sync="+str(s))
                plt.legend(loc="upper left", fontsize=5)
                plt.savefig(title+".png")
                plt.clf()


    #####################################
    # TIME RATIOS OVER HOG PER THREAD 
    #####################################
    for e in epoch_range:
        for r in rank_range:
            for b in batch_size_range:
                title = "Overall_Time_Ratios_Over_Hog_Per_Thread_Batch=%d_Epoch=%d_Rank=%d" % (b, e, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.ylabel("Hog Time / Cyclades Time")
                plt.xlabel("Thread")
                for s in sync_range:
                    for c in commands:
                        if 'hog' not in c:
                            baseline_hog_times = get_values(average_total_times, [hog_command], [e], [b], thread_range, [r], [s])
                            times = get_values(average_total_times, [c], [e], [b], thread_range, [r], [s])
                            ratio_times = [float(baseline_hog_times[i]) / float(times[i]) for i in range(len(times))]
                            plt.plot(thread_range, ratio_times, label=c+"_sync="+str(s))
                plt.legend(loc="upper left", fontsize=5)
                plt.savefig(title+".png")
                plt.clf()                                                            

    for e in epoch_range:
        for r in rank_range:
            for b in batch_size_range:
                title = "Gradient_Time_Ratios_Over_Hog_Per_Thread_Batch=%d_Epoch=%d_Rank=%d" % (b, e, r)
                plt.figure()
                plt.title(title, fontsize=12)
                plt.ylabel("Hog Time / Cyclades Time")
                plt.xlabel("Thread")
                for s in sync_range:
                    for c in commands:
                        if 'hog' not in c:
                            baseline_hog_times = get_values(average_total_times, [hog_command], [e], [b], thread_range, [r], [s])
                            times = get_values(average_gradient_times, [c], [e], [b], thread_range, [r], [s])
                            ratio_times = [float(baseline_hog_times[i]) / float(times[i]) for i in range(len(times))]
                            plt.plot(thread_range, ratio_times, label=c+"_sync="+str(s))
                plt.legend(loc="upper left", fontsize=5)
                plt.savefig(title+".png")
                plt.clf()                                                            


#draw_time_loss_graph(1, 200, [500], [1, 8, 16], [30], [0, 1], ["cyc_word_embeddings_cyc", "cyc_word_embeddings_hog"])
draw_time_loss_graph(0, 1000, [500], [8], [2], [1], ["cyc_graph_cuts_cyc", "cyc_graph_cuts_hog"])
#draw_epoch_loss_graph(0, 100, [300], [8], [2], [1], ["cyc_word_embeddings_cyc"], [.9])
    
