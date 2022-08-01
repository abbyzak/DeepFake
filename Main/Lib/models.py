from abc import ABCMeta, abstractmethod
import numpy as np
import cv2
from Lib.abs import Model

class OrthographicProjectionBlendshapes(Model):
    nParams = 6

    def __init__(self, nDlib_shape):
        self.nDlib_shape = nDlib_shape
        self.nParams += nDlib_shape

    def fun(self, x, params):

        s = params[0]

        r = params[1:4]

        t = params[4:6]
        w = params[6:]

        Dlib_mean3d = x[0]
        Dlib_shape = x[1]


        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape3D = Dlib_mean3d + np.sum(w[:, np.newaxis, np.newaxis] * Dlib_shape, axis=0)

        projected = s * np.dot(P, shape3D) + t[:, np.newaxis]

        return projected

    def jacobian(self, params, x, y):
        s = params[0]
        r = params[1:4]
        t = params[4:6]
        w = params[6:]

        Dlib_mean3d = x[0]
        Dlib_shape = x[1]

        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape3D = Dlib_mean3d + np.sum(w[:, np.newaxis, np.newaxis] * Dlib_shape, axis=0)

        nPoints = Dlib_mean3d.shape[1]
        

        jacobian = np.zeros((nPoints * 2, self.nParams))

        jacobian[:, 0] = np.dot(P, shape3D).flatten()

        stepSize = 10e-4
        step = np.zeros(self.nParams)
        step[1] = stepSize;
        jacobian[:, 1] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()
        step = np.zeros(self.nParams)
        step[2] = stepSize;
        jacobian[:, 2] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()
        step = np.zeros(self.nParams)
        step[3] = stepSize;
        jacobian[:, 3] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()

        jacobian[:nPoints, 4] = 1
        jacobian[nPoints:, 5] = 1

        startIdx = self.nParams - self.nDlib_shape
        for i in range(self.nDlib_shape):
            jacobian[:, i + startIdx] = s * np.dot(P, Dlib_shape[i]).flatten()

        return jacobian

    
    def getInitialParameters(self, x, y):
        Dlib_mean3d = x.T
        shape2D = y.T
   
        shape3DCentered = Dlib_mean3d - np.mean(Dlib_mean3d, axis=0)
        shape2DCentered = shape2D - np.mean(shape2D, axis=0)
    
        scale = np.linalg.norm(shape2DCentered) / np.linalg.norm(shape3DCentered[:, :2]) 
        t = np.mean(shape2D, axis=0) - np.mean(Dlib_mean3d[:, :2], axis=0)

        params = np.zeros(self.nParams)
        params[0] = scale
        params[4] = t[0]
        params[5] = t[1]

        return params
