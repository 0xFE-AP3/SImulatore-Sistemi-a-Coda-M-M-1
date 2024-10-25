import math
import random
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

VERBOSITY = 1  # 0 no output, 1 minimal, 2 debug

clock = 0
working_clocks = 0
pk = {}

# Funzioni di generazione pacchetti (solo standard)
def gen_function_exp(x):
    y = np.random.exponential(x)
    return 1 if y < 1 else round(y)

def gen_function_poi(x):
    y = np.random.poisson(x)
    return 1 if y < 1 else round(y)

# Definizione delle classi necessarie
class Stats:
    def __init__(self):
        self.ls_th = None
        self.lq_th = None
        self.ws_th = None
        self.wq_th = None
        self.rho_th = None
        self.ls_exp = 0
        self.lq_exp = 0
        self.ws_exp = 0
        self.wq_exp = 0
        self.rho_exp = 0
        self.ls_err = None
        self.lq_err = None
        self.ws_err = None
        self.wq_err = None
        self.rho_err = None
        self.num_pkt_generated = 0
        self.num_pkt_served = 0
        self.num_pkt_queue = 0
        self.num_pkt_server = 0
        self.num_working_clocks = 0

    def process_pkt_bin(self, pkt_bin):
        self.num_pkt_served = len(pkt_bin)
        for pkt in pkt_bin:
            self.ws_exp += pkt.life_server + pkt.life_queue
            self.wq_exp += pkt.life_queue
        self.ws_exp /= len(pkt_bin) if pkt_bin else 1
        self.wq_exp /= len(pkt_bin) if pkt_bin else 1

    def calculate_experimental(self):
        self.ls_exp = self.get_avg_pkt_system()
        self.lq_exp = self.get_avg_pkt_queue()
        self.rho_exp = working_clocks / clock if clock else 0

    def save_ideal(self, ls_th, lq_th, ws_th, wq_th, rho_th):
        self.ls_th = ls_th
        self.lq_th = lq_th
        self.ws_th = ws_th
        self.wq_th = wq_th
        self.rho_th = rho_th

    def calculate_errors(self):
        if self.rho_th:
            self.rho_err = (self.rho_exp - self.rho_th) / self.rho_th
        if self.ls_th:
            self.ls_err = (self.ls_exp - self.ls_th) / self.ls_th
        if self.lq_th:
            self.lq_err = (self.lq_exp - self.lq_th) / self.lq_th
        if self.ws_th:
            self.ws_err = (self.ws_exp - self.ws_th) / self.ws_th
        if self.wq_th:
            self.wq_err = (self.wq_exp - self.wq_th) / self.wq_th

    def get_avg_pkt_system(self):
        return (self.num_pkt_queue + self.num_pkt_server) / clock if clock else 0

    def get_avg_pkt_queue(self):
        return self.num_pkt_queue / clock if clock else 0

class Packet:
    def __init__(self, uid: int):
        self.uid = uid
        self.life_queue = 0
        self.life_server = 0

class Queue:
    def __init__(self):
        self.queue = []

    def tick(self):
        for pkt in self.queue:
            pkt.life_queue += 1

    def size(self):
        return len(self.queue)

    def empty(self):
        return self.size() < 1

    def extract(self):
        return self.queue.pop(0) if not self.empty() else None

    def enqueue(self, pkt: Packet):
        if pkt is not None:
            self.queue.append(pkt)

class Server:
    def __init__(self, avg_time):
        self.time_left = None
        self.avg_time = avg_time
        self.pkt = None
        self.pkt_bin = []

    def tick(self):
        if self.working():
            if self.time_left == 0:
                self.serve()
            else:
                self.time_left -= 1
                self.pkt.life_server += 1

    def serve(self):
        self.pkt_bin.append(self.pkt)
        self.pkt = None
        self.time_left = None

    def working(self):
        return self.pkt is not None

    def idle(self):
        return self.pkt is None

    def send(self, pkt):
        if self.idle() and pkt is not None:
            self.pkt = pkt
            self.time_left = gen_function_exp(self.avg_time)

class System:
    def __init__(self, queue: Queue, server: Server, stats: Stats):
        self.queue = queue
        self.server = server
        self.stats = stats

    def tick(self):
        if not self.queue.empty() and self.server.idle():
            new_pkt = self.queue.extract()
            self.server.send(new_pkt)

        self.queue.tick()
        self.server.tick()
        self.stats.num_pkt_queue += self.queue.size()
        self.stats.num_pkt_server += self.server.working()

    def send(self, pkt: Packet):
        if pkt is not None:
            self.queue.enqueue(pkt)

def progress_bar(index, total, title, bar_len=20, force_done=False):
    if VERBOSITY != 1:
        return
    percent_done = (index + 1) / total * 100
    done = round(percent_done / (100 / bar_len))
    print(f'\t⏳  {title}: [{"█" * int(done)}{"░" * (bar_len - done)}] {percent_done:.1f}%', end='\r')
    if force_done:
        print('\t✅')

# Esperimento standard
def experiment_standard(avg_gen_time, avg_ser_time, clock_max, seed) -> Stats:
    global clock, working_clocks, pk
    clock = 0
    working_clocks = 0
    pk = {}

    if VERBOSITY >= 1: print(f"Experiment rho={(1/avg_gen_time) / (1/avg_ser_time) * 100:.0f}%")
    np.random.seed(seed)
    if VERBOSITY >= 1: print("[I] Main: Seed set to", seed, end='\n\n')

    stats = Stats()
    q = Queue()
    s = Server(avg_ser_time)
    sys = System(q, s, stats)

    pkt_num = 1
    time_left = gen_function_poi(avg_gen_time)
    if VERBOSITY >= 2: print(f"[I] Main: generating first pkt_id={pkt_num} t_left={time_left}")

    while clock < clock_max:
        if clock % 5000 == 0: progress_bar(clock, clock_max, "Simulating")

        if time_left == 0:
            pkt = Packet(pkt_num)
            sys.send(pkt)
            pkt_num += 1
            time_left = gen_function_poi(avg_gen_time)
            if VERBOSITY >= 2: print(f"[I] Main: generating pkt_id={pkt_num} t_left={time_left}")

        sys.tick()
        tot_pkt = q.size() + s.working()
        pk[tot_pkt] = pk.get(tot_pkt, 0) + 1

        time_left -= 1
        clock += 1
        if s.working(): working_clocks += 1
    
    progress_bar(clock, clock_max, "Simulating", force_done=True)

    lambda_th = 1 / avg_gen_time
    mu_th = 1 / avg_ser_time
    rho_th = lambda_th / mu_th
    ls_th = rho_th / (1 - rho_th)
    lq_th = (lambda_th * lambda_th) / (mu_th * (mu_th - lambda_th))
    ws_th = 1 / (mu_th - lambda_th)
    wq_th = lambda_th / (mu_th * (mu_th - lambda_th))
    stats.save_ideal(ls_th, lq_th, ws_th, wq_th, rho_th)

    stats.process_pkt_bin(s.pkt_bin)
    stats.num_pkt_generated = pkt_num
    stats.num_working_clocks = working_clocks
    stats.calculate_experimental()
    stats.calculate_errors()
    
    print("\n> Packet stats")
    print("Packets total:", stats.num_pkt_generated)
    print("Working clock:", stats.num_working_clocks)
    print("Pk: { ", end="")
    for key, value in pk.items():
        pk[key] = value / clock_max * 100
        if int(key) <= 10: print(f"{str(key)}: {pk[key]:.2f}% | ", end = "")
    if len(pk) > 10: print(" ... }")
    else: print(" }")

    if VERBOSITY >= 1: print("\n> System stats")
    if VERBOSITY >= 1: print(stats)
    
    return stats

# Funzione principale che integra il batch
def main():
    # Configurazione esperimento
    avg_service = 15
    mode = 2
    seed = 53610
    clocks_to_simulate = 200_000
    mu_fixed = 1 / avg_service
    rhos = np.arange(0.06, 0.94, 0.01)

    # Creazione delle directory per risultati
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/errors", exist_ok=True)

    # Esecuzione di 4 esperimenti
    for i in range(1, 5):
        print(f"\nRunning experiment {i}...\n")
        ls_th_list, lq_th_list, ws_th_list, wq_th_list = [], [], [], []
        ls_exp_list, lq_exp_list, ws_exp_list, wq_exp_list = [], [], [], []
        ls_errs, lq_errs, ws_errs, wq_errs = [], [], [], []
        rho_ratios, generated_pkts, served_pkts = [], [], []

        for rho in rhos:
            lambda_calculated = mu_fixed * rho
            avg_generation = 1 / lambda_calculated

            print("\n----------------------------------------------------------\n")
            if VERBOSITY >= 1: print(f"Trying mu={mu_fixed:.3f} avg_ser={avg_service:.0f} rho={rho:.2f} --> avg_gen={1/lambda_calculated:.0f}")
            stats = experiment_standard(avg_generation, avg_service, clocks_to_simulate, seed)

            # Raccolta dati di statistica
            ls_exp_list.append(stats.ls_exp)
            lq_exp_list.append(stats.lq_exp)
            ls_errs.append(stats.ls_err)
            lq_errs.append(stats.lq_err)
            ls_th_list.append(stats.ls_th)
            lq_th_list.append(stats.lq_th)
            ws_exp_list.append(stats.ws_exp)
            wq_exp_list.append(stats.wq_exp)
            ws_errs.append(stats.ws_err)
            wq_errs.append(stats.wq_err)
            ws_th_list.append(stats.ws_th)
            wq_th_list.append(stats.wq_th)
            rho_ratios.append(rho / stats.rho_exp)
            generated_pkts.append(stats.num_pkt_generated)
            served_pkts.append(stats.num_pkt_served)

        # Salvataggio dei grafici
        plt.plot(rhos, rho_ratios, label="Ideal Rho / Experimental Rho")
        plt.legend(loc="best")
        plt.savefig(f'results/plots/rho-est-{seed}-{i}.png', bbox_inches='tight', dpi=200)
        plt.clf()

        plt.plot(rhos, generated_pkts, label="Number of pkt generated")
        plt.plot(rhos, served_pkts, label="Number of pkt served")
        plt.legend(loc="best")
        plt.savefig(f'results/plots/num-pkts-{seed}-{i}.png', bbox_inches='tight', dpi=200)
        plt.clf()

        plt.plot(rhos, ls_exp_list, label="Ls experimental")
        plt.plot(rhos, lq_exp_list, label="Lq experimental")
        plt.plot(rhos, ls_th_list, label="Ls ideal")
        plt.plot(rhos, lq_th_list, label="Lq ideal")
        plt.legend(loc="best")
        plt.savefig(f'results/plots/Lplot-{seed}-{i}.png', bbox_inches='tight', dpi=200)
        plt.clf()

        plt.plot(rhos, ws_exp_list, label="Ws experimental")
        plt.plot(rhos, wq_exp_list, label="Wq experimental")
        plt.plot(rhos, ws_th_list, label="Ws ideal")
        plt.plot(rhos, wq_th_list, label="Wq ideal")
        plt.legend(loc="best")
        plt.savefig(f'results/plots/Wplot-{seed}-{i}.png', bbox_inches='tight', dpi=200)
        plt.clf()

        plt.plot(rhos, ls_errs, label="Ls error (%)")
        plt.plot(rhos, lq_errs, label="Lq error (%)")
        plt.ylim(-100, 100)
        plt.legend(loc="best")
        plt.savefig(f'results/errors/Lerr-{seed}-{i}.png', bbox_inches='tight', dpi=200)
        plt.clf()

        plt.plot(rhos, ws_errs, label="Ws error (%)")
        plt.plot(rhos, wq_errs, label="Wq error (%)")
        plt.ylim(-100, 100)
        plt.legend(loc="best")
        plt.savefig(f'results/errors/Werr-{seed}-{i}.png', bbox_inches='tight', dpi=200)
        plt.clf()

    print("\nAll experiments completed and results saved.")

if __name__ == '__main__':
    main()
