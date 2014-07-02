import numpy as np
import numpy.ma as ma
from numpy import random
import matplotlib.pyplot as plt
from scipy import stats
import sde


def pairCorrelation(tEvents, tObserved, tau, masked=None):
    C = np.zeros((len(tau),))
    avgRate = len(tEvents)/tObserved[-1]
    for i, taui in enumerate(tau):
        # count events within tau of each event
        N = np.zeros((len(tEvents),))
        tauh = taui/2.
        for j, te in enumerate(tEvents):
            pobs = 1.
            if tauh > te:
                pobs -= (tauh - te)/taui
            if te > tObserved[-1]-tauh:
                pobs -= (tauh + te - tObserved[-1])/taui
            if masked is not None:
                sel = (tObserved > te - tauh &
                       tObserved < te + tauh)
                pnotmiss = np.sum(masked[sel])/np.sum(tObserved[sel])
            else:
                pnotmiss = 1.
            bias = pobs*pnotmiss
            N[j] = (float(len(tEvents[np.abs(te - tEvents) < tauh]))-1.)/bias
        C[i] = N.mean()/taui/avgRate
    return C


class PoissonProcess(object):
    def __init__(self, tau):
        self.tau = tau

    def nextEvent(self):
        return random.exponential(self.tau, 1)[0]

    def eventsByTime(self, maxT):
        events = []
        ti = 0.
        while ti < maxT:
            ti += self.nextEvent()
            if ti <= maxT:
                events.append(ti)
        return np.array(events)

    def eventTimeSeries(self, maxT, dt):
        te = self.eventsByTime(maxT)
        ie = np.round(te / dt).astype(int)
        ts = np.zeros((np.round(maxT / dt),))
        ts[ie] = 1
        return ts

    @staticmethod
    def fit(tEvents):
        dtEvents = np.diff(tEvents)
        tau = stats.expon.fit(dtEvents)[1]
        return tau


class OrnsteinUhlenbeckCoxProcess(object):
    def __init__(self, tau, logmu, sigma):
        self.logmu = logmu
        self.tau = tau
        self.sigma = sigma

    def getSDE(self):
        return sde.OrnsteinUhlenbeck(self.tau, self.logmu, self.sigma)

    def nextEvent(self, tau):
        return random.exponential(tau, 1)[0]

    def eventsByTime(self, maxT, dt=0.01):
        events = []
        logtauSDE = self.getSDE()
        t, logtau = logtauSDE.integrateEuler(0., maxT, dt, y0=self.logmu)
        ti = 0.
        while ti < maxT:
            tau = logtau[t>=ti][0]
            ti += self.nextEvent(tau)
            if ti <= maxT:
                events.append(ti)
        return np.array(events)

    def eventTimeSeries(self, maxT, dt):
        te = self.eventsByTime(maxT)
        ie = np.round(te / dt).astype(int)
        ts = np.zeros((np.round(maxT / dt),))
        ts[ie] = 1
        return ts
