import numpy as np
import cv2
import time
import yaml
import os


class ExperimentVideo(object):
    def __init__(self, path, outputName, notes={}, source=0, duration=60,
                 resolution=(640, 480), fps=11.5):
        self.path = path
        self.outputName = outputName
        self.source = source
        self.duration = duration
        self.notes = notes
        self.resolution = resolution
        self.fps = fps
        self.capture = None

    def startCapture(self):
        success, self.capture = cv2.VideoCapture(self.device)
        if not success:
            raise Exception('Error accessing camera.')
        else:
            print 'Capture started.'
        success = self.capture.set(cv2.cv.CV_CAP_PROP_FPS, self.fps)
        if not success:
            print 'Unable to set frame rate.'
        success = self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,
                                   self.resolution[0])
        if not success:
            print 'Unable to set frame width.'
        success = self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,
                                   self.resolution[1])
        if not success:
            print 'Unable to set frame height.'

    def runExperiment(self):
        nFrames = self.fps * self.duration * 60.
        frameT = []
        videoOut = cv2.VideoWriter()  # raw output
        name = os.path.join(self.path, self.outputName + '.avi')
        success = videoOut.open(name, 0)
        if not success:
            raise Exception('Unable to open output video.')
        self.startCapture()
        success = True
        framesCollected = 0
        while success and framesCollected < nFrames:
            succcess, frame = self.capture.read()
            if success:
                frameT.append(time.clock())
                framesCollected += 1
                cv2.imshow('Live Capture', frame)
                wsuccess = videoOut.write(frame)
                if not wsuccess:
                    raise Exception('Unable to write to output video.')
        cv2.destroyWindow('Live Capture')
        self.stopCapture()
        self.saveNotes()

    def stopCapture(self):
        self.capture.release()
        print 'Capture stopped.'

    def saveNotes(self):
        output = {'path': self.path,
                  'videoFile': self.outputName + '.avi',
                  'duration': self.duration,
                  'resolution': self.resolution,
                  'fps': self.fps,
                  'notes': self.notes}
        name = os.path.join(self.path, self.outputName + '.yaml')
        with open(name, 'w') as f:
            yaml.dump(output, f)
        print 'Experiment notes saved.'
