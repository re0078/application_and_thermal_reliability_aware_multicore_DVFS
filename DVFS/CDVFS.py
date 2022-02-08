"""
CDVFS.py

Change core frequencies according to a RL-based proactive algorithm.
"""
import sys, os, sim
import numpy as np


class CDVFS(object):
    def setup(self, args):
        self.events = []
        args = args.split(':')
        for i in range(0, len(args), 3):
            self.events.append((int(args[i]) * sim.util.Time.NS, int(args[i + 1]), int(args[i + 2])))
        self.events.sort()
        sim.util.Every(100 * sim.util.Time.NS, self.periodic, roi_only=True)

    def periodic(self, time, time_delta):
        while self.events and time >= self.events[0][0]:
            t, cpu, freq = self.events[0]
            self.events = self.events[1:]
            sim.dvfs.set_frequency(cpu, freq)

    def get_freq(self):
        freqs = np.empty_like(self.events)
        for i, event in enumerate(self.events):
            t, cpu, _ = event
            freqs[i] = sim.dvfs.get_frequencies(cpu)
        return freqs

    def set_freq(self, freqs):
        for i in range(len(self.events)):
            self.events[i] = self.events[i][0], self.events[i][1], freqs[i]
            t, cpu, freq = self.events[i]
            sim.dvfs.set_frequency(cpu, freq)

    def get_t(self):
        t = np.empty_like(self.events)
        for i, event in enumerate(self.events):
            _, cpu, _ = event
            t[i] = sim.dvfs.get_frequencies(cpu)
        return t

    def get_p(self):
        return np.array([sim.dvfs.get_power(cpu) for _, cpu, _ in self.events])

cdvfs = CDVFS()
sim.util.register(cdvfs)