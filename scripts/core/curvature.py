import math
import numpy as np
from scipy.stats import vonmises

import cv2

class VonMisesBuffer(object):

    def __init__(self, k=3):
        self.vonmises_range = np.arange(0, math.pi * 2, 0.0001)
        self.vonmises_base = vonmises(k)
        self.vonmises_values = self.vonmises_base.pdf(self.vonmises_range)

    def pdf(self, x):

        i = int(math.fabs(x) * 10000)
        if i >= 0 and i < len(self.vonmises_values):
            return self.vonmises_values[i]
        return 0.0


class CurvaturePredictor(object):

    def __init__(self, ariadne):
        self.ariadne = ariadne

    def computeScore(self, path, initial_direction=None):
        return 1.0


class CurvatureSimplePredictor(CurvaturePredictor):

    def __init__(self, ariadne, kappa = 7, max_angle=math.pi * 0.5):
        super(CurvatureSimplePredictor, self).__init__(ariadne)
        self.max_angle = max_angle
        self.vonmises = VonMisesBuffer(k=kappa)

    def computeScore(self, path, debug=False):
       
        #######################################
        # Degenerate Paths
        #######################################
        if len(path) <= 2:
            return 1.0

        #######################################
        # Normal Path
        #######################################
        directions = []
        points = self.ariadne.graph.get2DPoints(path)
        for i in range(len(points) - 2, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)


        try:
            angle = math.acos(np.dot(directions[0], directions[1]))
            print("angle", angle)
            if angle > self.max_angle:
                print("max angle reached!")
                angle = 100.0
        except:
            angle = 0.0
    
        return self.vonmises.pdf(angle)



class CurvatureVonMisesLastPredictor(CurvaturePredictor):

    def __init__(self, ariadne, kappa=4, max_angle=math.pi * 0.5):
        super(CurvatureVonMisesLastPredictor, self).__init__(ariadne)
        self.vonmises = VonMisesBuffer(k=kappa)
        self.max_angle = max_angle

    def computeScore(self, path):


        #######################################
        # Single Node Path
        #######################################
        if len(path) <= 2:
            return 1.0


        #######################################
        # Normal Path
        #######################################
        directions = []
        points = self.ariadne.graph.get2DPoints(path)
        for i in range(len(points) - 2, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

        try:
            angle = math.acos(np.dot(directions[0], directions[1]))
        except:
            angle = 0.0

        prob = self.vonmises.pdf(angle)
        prob = 1 if prob > 1 else prob
        return prob




class CurvatureVonMisesPredictor(CurvaturePredictor):

    def __init__(self, ariadne, kappa=8, max_angle=math.pi * 0.5):
        super(CurvatureVonMisesPredictor, self).__init__(ariadne)
        self.vonmises = VonMisesBuffer(k=kappa)
        self.max_angle = max_angle

    def computeScore(self, path, debug=False):

        #######################################
        # Single Node Path
        #######################################
        if len(path) <= 2:
            return 1.0

        #######################################
        # Normal Path
        #######################################
        directions = []
        points = self.ariadne.graph.get2DPoints(path)
        for i in range(1, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

        thetas = []
        for i in range(1, len(directions)):
            d1 = directions[i - 1]
            d2 = directions[i]
            a1 = math.atan2(d1[1], d1[0])
            a2 = math.atan2(d2[1], d2[0])
            angle = a1 - a2
            angle = math.acos(math.cos(angle))

            thetas.append(angle)

        #print("thetas: ", thetas)
        if math.fabs(thetas[-1]) > self.max_angle:
            return 0.0

        if len(thetas) == 1:
            prob =  self.vonmises.pdf(thetas[0])
        elif len(thetas) > 1:
            probs = [self.vonmises.pdf(theta) for theta in thetas]
            #print("probs: ", probs)
            prob = np.prod(probs)
            '''
            posterios = []
            for i in range(1, len(thetas)):
                t1 = thetas[i - 1]
                t2 = thetas[i]
                posterios.append(self.vonmises.pdf(t1 - t2))

            prob =  np.prod(np.array(posterios).ravel())**(1.0 / (len(points) - 3.0))
            '''
        
        #prob = 1 if prob > 1 else prob
        return prob

