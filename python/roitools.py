import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2


class ThresholdedImageSelector(object):
    def __init__(self, im):
        self.ax = plt.gca()
        self.im = im
        self.x = None
        self.y = None
        self.im = im
        self.implot = plt.imshow(im)
        self.regions = self.getRegions()
        self.ax.figure.canvas.mpl_connect('button_press_event',
                                         self.on_click)
        self.draw()

    def getRegions(self):
        contours, hierarchy = cv2.findContours(
            np.uint8(self.im), cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def getClickedRegion(self, contours):
        if (self.x is None) or (self.y is None):
            return None
        xy = (int(self.x), int(self.y))
        inside = np.array([cv2.pointPolygonTest(contour, xy, False)
                           for contour in contours])
        selRegion = inside >= 0.
        if np.any(selRegion):
            return contours[selRegion.nonzero()[0]]

    def fillSelectedRegion(self, contour):
        im = cv2.cvtColor(self.im.astype('uint8')*255,
                          cv2.COLOR_GRAY2RGB)
        if contour is not None:
            cv2.drawContours(im,
                             [contour],
                             0, (255, 0, 0), -1)
        self.implot.set_data(im)

    def getImageNonSelectedRegions(self):
        im = cv2.cvtColor(self.im.astype('uint8')*255,
                          cv2.COLOR_GRAY2RGB)
        contour = self.getClickedRegion(self.regions)
        if contour is not None:
            cv2.drawContours(im,
                             [contour],
                             0, (0, 0, 0), -1)
        return im[:, :, 0].squeeze()

    def on_click(self, event):
        if event.inaxes:
            self.x = event.xdata
            self.y = event.ydata
        self.draw()

    def draw(self):
        clicked = self.getClickedRegion(self.regions)
        self.fillSelectedRegion(clicked)
        self.ax.figure.canvas.draw()


# modified from http://stackoverflow.com/questions/12052379/ ...
# matplotlib-draw-a-selection-area-in-the-shape-of-a-rectangle-with-the-mouse


class RectangleRegionSelector(object):
    isPressed = False

    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0, 0), 1, 1, color='b', fill=False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.on_motion)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.isPressed = True

    def on_motion(self, event):
        if self.isPressed:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_rect()

    def on_release(self, event):
        self.isPressed = False
        if event.xdata is not None and event.ydata is not None:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_rect()
        else:
            print "Mouse must be released within the axes, try again."

    def draw_rect(self):
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

    def asXYWH(self):
        return (min(self.x0, self.x1), min(self.y0, self.y1),
                abs(self.x1-self.x0), abs(self.y1-self.y0))


class CircleRegionSelector(object):
    isPressed = False

    def __init__(self):
        self.ax = plt.gca()
        self.circle = plt.Circle((0, 0), radius=10, color='r', fill=False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.circle)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.on_motion)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.isPressed = True

    def on_motion(self, event):
        if self.isPressed:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_circle()

    def on_release(self, event):
        self.isPressed = False
        if event.xdata is not None and event.ydata is not None:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_circle()
        else:
            print "Mouse must be released within the axes, try again."

    def draw_circle(self):
        self.circle.center = (self.x0, self.y0)
        self.circle.radius = np.sqrt((self.x1-self.x0)**2 +
                                     (self.y1-self.y0)**2)
        self.ax.figure.canvas.draw()

    def asXYR(self):
        return (self.circle.center[0], self.circle.center[1],
                self.circle.radius)


class LineRegionSelector(object):
    isPressed = False

    def __init__(self):
        self.ax = plt.gca()
        self.line = plt.Line2D((0, 0), (1, 1), linewidth=2, color='r')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_line(self.line)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.on_motion)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.isPressed = True

    def on_motion(self, event):
        if self.isPressed:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_line()

    def on_release(self, event):
        self.isPressed = False
        if event.xdata is not None and event.ydata is not None:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_line()
        else:
            print "Mouse must be released within the axes, try again."

    def draw_line(self):
        self.line.set_xdata((self.x0, self.x1))
        self.line.set_ydata((self.y0, self.y1))
        self.ax.figure.canvas.draw()

    def distance(self):
        return np.sqrt((self.x1-self.x0)**2 +
                       (self.y1-self.y0)**2)