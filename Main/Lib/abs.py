from abc import ABCMeta, abstractmethod
import numpy as np
import cv2

class Model:
    __metaclass__ = ABCMeta

    nParams = 0


    def residual(self, params, x, y):
        r = y - self.fun(x, params)
        r = r.flatten()

        return r


    @abstractmethod
    def fun(self, x, params):
        pass

    @abstractmethod
    def jacobian(self, params, x, y):
        pass



    @abstractmethod
    def getInitialParameters(self):
        pass
