import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import itertools
from collections import MutableSequence
import cv2
import wormtracker.wormimageprocessor as wip
import multiprocessing as multi
from numba import jit
import scipy.stats as ss
from tsstats import *
from stats import *


def configureMatplotLibStyle():
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.labelsize'] = 'x-large'
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['grid.color'] = (0.5, 0.5, 0.5)
    mpl.rcParams['legend.fontsize'] = 'medium'
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.frameon'] = False


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


class WormTrajectory:
    filterByWidth = False  # static variable!

    def __init__(self, h5obj, strain, wormID, videoFilePath=None):
        self.firstFrame = None
        self.h5obj = h5obj
        self.h5ref = self.h5obj['worms'][strain][wormID]
        self.strain = strain
        self.wormID = wormID
        self.frameRate = self.h5obj['/video/frameRate'][0]
        self.pixelsPerMicron = self.h5obj['/video/pixelsPerMicron'][0]
        self.foodCircle = self.h5ref['foodCircle'][...] / self.pixelsPerMicron
        self.t = self.h5ref['time']
        self.maxFrameNumber = self.t.shape[0]
        self.X = self.h5ref['X']
        self.Xhead = self.h5ref['Xhead']
        self.v = self.h5ref['v']
        self.s = self.h5ref['s']
        self.phi = self.h5ref['phi']
        self.psi = self.h5ref['psi']
        self.dpsi = self.h5ref['dpsi']
        self.Ctheta = self.h5ref['Ctheta']
        self.ltheta = self.h5ref['ltheta']
        self.vtheta = self.h5ref['vtheta']
        self.length = self.h5ref['avgLength']
        self.width = self.h5ref['avgWidth']
        self.badFrames = self.h5ref['badFrames'][...]
        self.allCentroidMissing = np.all(self.badFrames)

        if videoFilePath is not None:
            videoFile = self.h5obj['/video/videoFile'][0]
            self.videoFile = os.path.join(videoFilePath, videoFile)
        else:
            self.videoFile = None

        self.skeleton = self.h5ref['skeletonSpline']
        self.posture = self.h5ref['posture']
        self.orientationFixed = self.h5ref['orientationFixed'][...]
        self.allPostureMissing = np.all(np.logical_not(self.orientationFixed))

    def readFirstFrame(self):
        if self.videoFile is None:
            self.firstFrame = None
            return
        video = cv2.VideoCapture()
        if video.open(self.videoFile):
            success, firstFrame = video.read()
            if not success:
                raise Exception("Couldn't read video")
            else:
                firstFrameChannels = cv2.split(firstFrame)
                frame = firstFrameChannels[0]
                crop = self.h5ref['cropRegion'][...]
                frame = wip.cropImageToRegion(frame, crop)
                ip = self.getImageProcessor()
                frame = ip.applyBackgroundFilter(frame)
                self.firstFrame = cv2.normalize(frame,
                                                alpha=0,
                                                beta=255,
                                                norm_type=cv2.NORM_MINMAX)
        else:
            raise Exception("Couldn't open video")

    def getImageProcessor(self):
        ip = wip.WormImageProcessor()
        ip.backgroundDiskRadius = \
            self.h5obj['/video/backgroundDiskRadius'][0]
        ip.pixelSize = self.h5obj['/video/pixelsPerMicron'][0]
        ip.threshold = self.h5obj['/video/threshold'][0]
        ip.wormDiskRadius = self.h5obj['/video/wormDiskRadius'][0]
        ip.wormAreaThresholdRange = \
            self.h5obj['/video/wormAreaThresholdRange'][...]
        # length
        # width
        return ip

    def getMaskedCentroid(self, data):
        data = ma.array(data)
        sel = self.badFrames
        data[sel, ...] = ma.masked
        data[np.isnan(data)] = ma.masked
        return data

    def getMaskedPosture(self, data):
        data = ma.array(data)
        sel = np.logical_or(np.logical_not(self.orientationFixed),
                            self.badFrames)
        data[sel, ...] = ma.masked
        data[np.isnan(data)] = ma.masked
        return data

    def plotTrajectory(self, color='k', showFrame=True, showPlot=True):
        if showFrame and self.firstFrame is not None:
            plt.imshow(self.firstFrame, plt.gray(),
                       origin='lower',
                       extent=(0,
                               self.firstFrame.shape[1]/self.pixelsPerMicron,
                               0,
                               self.firstFrame.shape[0]/self.pixelsPerMicron))
            plt.hold(True)
        X = self.getMaskedCentroid(self.X)
        plt.plot(X[:, 0], X[:, 1], '-', color=color)
        plt.hold(True)
        if self.foodCircle is not None:
            circle = plt.Circle(self.foodCircle[0:2],
                                radius=self.foodCircle[-1],
                                color='r', fill=False)
            plt.gca().add_patch(circle)
        plt.xlim((0, 10000))
        plt.ylim((0, 10000))
        plt.xlabel('x (um)')
        plt.xlabel('y (um)')
        plt.gca().set_aspect('equal')
        if showPlot:
            plt.show()

    def plotSpeed(self, showPlot=True):
        s = self.getMaskedCentroid(self.s)
        plt.plot(self.t, s, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (um/s)')
        if showPlot:
            plt.show()

    def plotBearing(self, showPlot=True):
        phi = self.getMaskedCentroid(self.phi)
        plt.plot(self.t, phi/np.pi, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Bearing ($\pi$ rad)')
        if showPlot:
            plt.show()

    def getSpeedDistribution(self, bins=None, useKernelDensity=True):
        if bins is None:
            bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))

        s = self.getMaskedCentroid(self.s)
        if not useKernelDensity:
            out = np.histogram(s.compressed(), bins,
                               density=True)[0]
        else:
            kd = ss.gaussian_kde(self.getMaskedCentroid(self.s).compressed())
            out = kd.evaluate(bins)
        return out

    def plotSpeedDistribution(self, bins=None, color='k', showPlot=True):
        if bins is None:
            bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))
        s = self.getMaskedCentroid(self.s)
        plt.hist(s.compressed(), bins, normed=True, facecolor=color)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        if showPlot:
            plt.show()

    def getSpeedAutocorrelation(self, maxT=100):
        n = int(np.round(maxT*self.frameRate))
        tau = range(n)/self.frameRate
        s = self.getMaskedCentroid(self.s)
        C = acf(s, n)
        return tau, C

    def plotSpeedAutocorrelation(self, maxT=100, color='k', showPlot=True):
        tau, C = self.getSpeedAutocorrelation(maxT)
        plt.plot(tau, C, '-', color=color)
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')
        if showPlot:
            plt.show()

    def getBearingAutocorrelation(self, maxT=100):
        n = int(np.round(maxT*self.frameRate))
        tau = range(n)/self.frameRate
        psi = self.getMaskedPosture(self.psi)
        C = circacf(psi, n)
        return tau, C

    def plotBearingAutocorrelation(self, maxT=100, color='k', showPlot=True):
        tau, C = self.getBearingAutocorrelation(maxT)
        plt.semilogx(tau, C, '-', color=color)
        plt.xlabel(r'$\log \tau / (s)$')
        plt.ylabel(r'$\langle \cos\left[\psi(t)-\psi(t+\tau)\right]\rangle$')
        if showPlot:
            plt.show()

    def getMeanSquaredDisplacement(self, tau=None):
        if tau is None:
            tau = np.logspace(-1, 3, 200)

        lags = np.round(tau*self.frameRate)
        Sigma = ma.zeros((200,))
        X = self.getMaskedCentroid(self.X)
        for i, lag in enumerate(lags):
            displacements = X[lag:, :] - X[:-lag, :]
            Sigma[i] = np.mean(np.log10(np.sum(displacements**2, axis=1)))

        return (tau, Sigma)

    def plotMeanSquaredDisplacement(self, tau=None, showPlot=True):
        tau, Sigma = self.getMeanSquaredDisplacement(tau)
        plt.plot(np.log10(tau), Sigma, 'k.')
        plt.xlabel(r'log $\tau$ \ (s)')
        plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
        if showPlot:
            plt.show()

    def plotPosturalCovariance(self, showPlot=True):
        if self.Ctheta is None:
            return
        plt.imshow(self.Ctheta, plt.get_cmap('PuOr'))
        plt.clim((-0.5, 0.5))
        plt.colorbar()
        plt.grid(False)
        if showPlot:
            plt.show()

    def plotPosturalModeDistribution(self, color='k', showPlot=True):
        if self.Ctheta is None:
            return
        plt.plot(np.cumsum(self.ltheta)/np.sum(self.ltheta), '.-', color=color,
                 label='{0} {1}'.format(self.strain, self.wormID))
        plt.xlabel('Postural Mode')
        plt.ylabel('%% Variance')
        plt.ylim((0, 1))
        if showPlot:
            plt.show()

    def plotPosturalTimeSeries(self, postureVec, color='k', showPlot=True):
        if self.Ctheta is None:
            return
        if isinstance(postureVec, int):
            postureVec = self.vtheta[:, postureVec]
        posture = self.getMaskedPosture(self.posture)
        A = np.dot(posture, postureVec)
        missing = np.any(posture.mask, axis=1)
        A[missing] = ma.masked
        plt.plot(self.t, A, '.', color=color,
                 label='{0} {1}'.format(self.strain, self.wormID))
        if showPlot:
            plt.show()

    def plotPosturalPhaseSpace(self, postureVec1, postureVec2, color='k',
                               showPlot=True):
        if self.Ctheta is None:
            return
        if isinstance(postureVec1, int):
            postureVec1 = self.vtheta[:, postureVec1]
        if isinstance(postureVec2, int):
            postureVec2 = self.vtheta[:, postureVec2]
        posture = self.getMaskedPosture(self.posture)
        missing = np.any(posture.mask, axis=1)
        A = np.dot(posture, postureVec1)
        A[missing] = ma.masked
        B = np.dot(posture, postureVec2)
        B[missing] = ma.masked
        plt.scatter(A, B, marker='.', c=color, s=5)
        if showPlot:
            plt.show()


class WormTrajectoryEnsemble(MutableSequence):
    def __init__(self, trajectoryIter, name=None, nameFunc=None):
        MutableSequence.__init__(self, trajectoryIter)
        self.name = name
        if nameFunc is None:
            nameFunc = lambda t: t.strain + ' ' + t.wormID
        self.nameFunc = nameFunc

    def _from_iterable(self, iter):
        return WormTrajectoryEnsemble(iter, name=self.name,
                                      nameFunc=self.nameFunc)

    def sort(self, cmp=None, key=None):
        if cmp is None and key is None:
            key = lambda t: int(t.wormID)
        MutableSequence.sort(self, cmp=cmp, key=key)

    def splitByStrain(self):
        strains = set([traj.strain for traj in self])
        return {strain: WormTrajectoryEnsemble([traj for traj in self
                                                if traj.strain == strain])
                for strain in strains}

    def readFirstFrameAll(self):
        for t in self:
            t.readFirstFrame()

    def calculatePosturalMeasurements(self):
        posture = []
        for traj in self:
            posturea = traj.getMaskedPosture(traj.posture)
            missing = np.any(posturea.mask, axis=1)
            if np.all(missing):
                continue
            else:
                posture.append(posturea[~missing, :])
        if len(posture) > 0:
            posture = np.concatenate(posture).T
            self.Ctheta = np.cov(posture)
            self.ltheta, self.vtheta = LA.eig(self.Ctheta)
        else:
            self.Ctheta = None
            self.ltheta = None
            self.vtheta = None

    def ensembleAverage(self, compFunc, nSamples=1000):
        samples = np.array([compFunc(traj) for traj in self])
        return bootstrap(samples, nSamples)

    def tilePlots(self, plotFunc, ni=4, nj=4):
        for i, t in enumerate(self):
            plt.subplot(ni, nj, i+1)
            plotFunc(t)
            plt.title(self.nameFunc(t))
        plt.show()

    def getMedianSpeed(self):
        func = lambda t: np.array([ma.median(t.getMaskedCentroid(t.s))])
        return self.ensembleAverage(func)

    def getMeanSpeed(self):
        func = lambda t: np.array([ma.mean(t.getMaskedCentroid(t.s))])
        return self.ensembleAverage(func)

    def plotSpeedDistribution(self, bins=None, color='k', showPlot=True):
        if bins is None:
            bins = np.linspace(0, 500, 200)
        p, pl, pu = self.ensembleAverage(lambda x: x.getSpeedDistribution(bins))
        plt.plot(bins, p, '.-', color=color, label=self.name)
        plt.hold(True)
        plt.fill_between(bins, pl, pu, facecolor=color, alpha=0.3)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.xlim([0,max(bins)])
        plt.grid(True)
        if showPlot:
            plt.show()

    def plotSpeedDistributions(self, bins=None, showPlot=True):
        if bins is None:
            bins = np.linspace(0, 500, 200)
        for i, t in enumerate(self):
            color = plt.cm.jet(float(i)/float(len(self)-1))
            p = t.getSpeedDistribution(bins)[0]
            plt.plot(bins,p,'.-',color=color,
                     label=self.nameFunc(t))
            plt.hold(True)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.xlim([0, max(bins)])
        plt.grid(True)
        plt.legend()
        if showPlot:
            plt.show()

    def plotSpeedAutocorrelation(self, maxT=100, color='k', showPlot=True):
        # assume all same frame rate
        n = int(np.round(maxT*self[0].frameRate))
        tau = range(n)/self[0].frameRate
        C, Cl, Cu = self.ensembleAverage(lambda x: x.getSpeedAutocorrelation(maxT)[1])
        plt.plot(tau, C, '.-', color=color, label=self.name)
        plt.fill_between(tau, Cl, Cu, facecolor=color, alpha=0.3)
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')
        plt.grid(True)
        # TODO: smart xlim
        if showPlot:
            plt.show()

    def plotBearingAutocorrelation(self, maxT=100, color='k', showPlot=True):
        # assume all same frame rate
        n = int(np.round(maxT*self[0].frameRate))
        tau = range(n)/self[0].frameRate
        C, Cl, Cu = self.ensembleAverage(lambda x: x.getBearingAutocorrelation(maxT)[1])
        plt.semilogx(tau, C, '.-', color=color, label=self.name)
        plt.fill_between(tau, Cl, Cu, facecolor=color, alpha=0.3)
        plt.xlabel(r'$\log \tau / (s)$')
        plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')
        plt.grid(True)
        # TODO: smart xlim
        if showPlot:
            plt.show()

    def plotMeanSquaredDisplacement(self, tau=None, color='k', showPlot=True):
        tau = np.logspace(-1,3,200)
        S, Sl, Su = self.ensembleAverage(lambda x: x.getMeanSquaredDisplacement(tau)[1])
        log_tau = np.log10(tau)
        plt.plot(log_tau, S, '.-', color=color, label=self.name)
        plt.hold(True)
        plt.fill_between(log_tau, Sl, Su, facecolor=color, alpha=0.3)
        plt.xlabel(r'log $\tau$ \ (s)')
        plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
        plt.xlim((np.min(np.log10(tau)), np.max(np.log10(tau))))
        plt.grid(True)
        if showPlot:
            plt.show()

    def plotPosturalCovariance(self, showPlot=True):
        if self.Ctheta is None:
            return
        plt.imshow(self.Ctheta, plt.get_cmap('PuOr'))
        plt.clim((-0.5, 0.5))
        plt.colorbar()
        if showPlot:
            plt.show()

    def plotPosturalModeDistribution(self, color='k', showPlot=True):
        if self.Ctheta is None:
            return
        plt.plot(np.cumsum(self.ltheta)/np.sum(self.ltheta), '.-', color=color)
        plt.xlabel('Postural Mode')
        plt.ylabel('%% Variance')
        plt.ylim((0, 1))
        if showPlot:
            plt.show()

    def plotAveragePosturalModeDistribution(self, color='k', showPlot=True):
        l, ll, lu = self.ensembleAverage(lambda x: np.cumsum(x.ltheta) / np.sum(x.ltheta))
        plt.plot(l, '.-', color=color)
        plt.hold(True)
        plt.fill_between(xrange(l.shape[0]), ll, lu,
                         facecolor=color, alpha=0.3)
        plt.xlabel('Postural Mode')
        plt.ylabel('%% Variance')
        plt.ylim((0, 1))
        if showPlot:
            plt.show()

    def plotPosturalPhaseSpaceDensity(self, postureVec1, postureVec2,
                                      showPlot=True):
        if self.Ctheta is None:
            return
        if isinstance(postureVec1, int):
            postureVec1 = self.vtheta[:, postureVec1]
        if isinstance(postureVec2, int):
                postureVec2 = self.vtheta[:, postureVec2]
        A = []
        B = []
        for traj in self:
            if traj.Ctheta is None:
                continue
            posture = traj.getMaskedPosture(traj.posture)
            missing = np.any(posture.mask, axis=1)
            a = np.dot(posture, postureVec1)
            a[missing] = ma.masked
            A.append(a)
            b = np.dot(posture, postureVec2)
            b[missing] = ma.masked
            B.append(b)
        plt.hexbin(np.concatenate(A), np.concatenate(B), bins='log')
        if showPlot:
            plt.show()


class WormTrajectoryEnsembleGroup(MutableSequence):
    def __init__(self, ensembles, name=None, colorScheme=None):
        MutableSequence.__init__(self, ensembles)
        self.name = name
        if colorScheme is None:
            self.colorScheme = lambda e: 'k'
        else:
            self.colorScheme = colorScheme

    def _from_iterable(self, iter):
        return WormTrajectoryEnsembleGroup(iter, name=self.name,
                                           colorScheme=self.colorScheme)

    def sort(self, cmp=None, key=None):
        if cmp is None and key is None:
            key = lambda e: e.name
        MutableSequence.sort(self, cmp=cmp, key=key)

    def calculatePosturalMeasurements(self):
        for ens in self:
            ens.calculatePosturalMeasurements()

    def tilePlots(self, plotFunc, ni=1, nj=None, showPlot=True):
        if nj is None:
            nj = np.ceil(len(self)/ni)
        if ni is None:
            ni = np.ceil(len(self)/nj)
        for i, e in enumerate(self):
            plt.subplot(ni, nj, i+1)
            plotFunc(e)
            plt.title(e.name)
        plt.show()

    def plotSpeedDistribution(self, bins=None, showPlot=True):
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotSpeedDistribution(bins=bins,
                                      color=color,
                                      showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotSpeedAutocorrelation(self, showPlot=True):
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotSpeedAutocorrelation(color=color,
                                         showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotMeanSquaredDisplacement(self, tau=None, showPlot=True):
        if tau is None:
            tau = np.logspace(-1,3,80)
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotMeanSquaredDisplacement(color=color,
                                            showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotPosturalModeDistribution(self, showPlot=True):
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotAveragePosturalModeDistribution(color=color,
                                                    showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()
