from math import sqrt, tan, radians
import numpy as np
import math_Tools as tl

# @param rLE :  (leading edge radius)
# @param Xup : (upper crest abscissa)
# @param Yup : (upper crest ordinate)
# @param YXXup : (upper crest curvature)
# @param Xlow : (lower crest abscissa)
# @param Ylow : (lower crest ordinate)
# @param YXXlow : (lower crest curvature)
# @param yTE : (trailing edge ordinate)
# @param deltaYTE :  (trailing edge thickness)
# @param alphaTE : (trailing edge direction)
# @param betaTE : (trailing edge wedge angle)
class airfoil(object):
    def __init__(self, rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE):
        self.rLE = rLE
        self.Xup = Xup
        self.Yup = Yup
        self.YXXup = YXXup
        self.Xlow = Xlow
        self.Ylow = Ylow
        self.YXXlow = YXXlow
        self.yTE = yTE
        self.deltaYTE = deltaYTE
        self.alphaTE = alphaTE
        self.betaTE = betaTE

        self.AUp = []
        self.ALow = []

    @property
    def XCoordinates(self):
        # # Changed to 11 after discussion to accomodate model input
        # return np.linspace(0, 1, 11)
        # return np.linspace(0, 1, 10)
        # return np.linspace(0, 1, 5)
        return np.linspace(0, 1, 15)

    @property
    def YCoordinatesUpper(self):
        xCoor = self.XCoordinates
        yCoorUpper = []
        for i in range(len(xCoor)):
            z_upper = 0
            for n in range(1, 7):
                term = self.AUp[n - 1] * xCoor[i] ** (n - 0.5)
                z_upper += term
            yCoorUpper.append(z_upper)

        return yCoorUpper

    @property
    def YCoordinatesLower(self):
        xCoor = self.XCoordinates
        yCoorLower = []
        for i in range(len(xCoor)):
            z_upper = 0
            for n in range(1, 7):
                term = self.ALow[n - 1] * xCoor[i] ** (n - 0.5)
                z_upper += term
            yCoorLower.append(z_upper)
        return yCoorLower

    def build(self):

        def CUp():
            c1 = [1, 1, 1, 1, 1, 1]
            c2 = [self.Xup ** (1 / 2), self.Xup ** (3 / 2), self.Xup ** (5 / 2), self.Xup ** (7 / 2),
                  self.Xup ** (9 / 2), self.Xup ** (11 / 2)]
            c3 = [1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2]
            c4 = [(1 / 2) * self.Xup ** (-1 / 2), (3 / 2) * self.Xup ** (1 / 2), (5 / 2) * self.Xup ** (3 / 2),
                  (7 / 2) * self.Xup ** (5 / 2), (9 / 2) * self.Xup ** (7 / 2), (11 / 2) * self.Xup ** (9 / 2)]
            c5 = [(-1 / 4) * self.Xup ** (-3 / 2), (3 / 4) * self.Xup ** (-1 / 2), (15 / 4) * self.Xup ** (1 / 2),
                  (15 / 4) * self.Xup ** (3 / 2), (63 / 4) * self.Xup ** (5 / 2), (99 / 4) * self.Xup ** (7 / 2)]
            c6 = [1, 0, 0, 0, 0, 0]
            Cup = [c1, c2, c3, c4, c5, c6]

            return Cup

        def CLow():
            c7 = [1, 1, 1, 1, 1, 1]
            c8 = [self.Xlow ** (1 / 2), self.Xlow ** (3 / 2), self.Xlow ** (5 / 2), self.Xlow ** (7 / 2),
                  self.Xlow ** (9 / 2), self.Xlow ** (11 / 2)]
            c9 = [1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2]
            c10 = [(1 / 2) * self.Xlow ** (-1 / 2), (3 / 2) * self.Xlow ** (1 / 2), (5 / 2) * self.Xlow ** (3 / 2),
                   (7 / 2) * self.Xlow ** (5 / 2), (9 / 2) * self.Xlow ** (7 / 2), (11 / 2) * self.Xlow ** (9 / 2)]
            c11 = [(-1 / 4) * self.Xlow ** (-3 / 2), (3 / 4) * self.Xlow ** (-1 / 2), (15 / 4) * self.Xlow ** (1 / 2),
                   (15 / 4) * self.Xlow ** (3 / 2), (63 / 4) * self.Xlow ** (5 / 2), (99 / 4) * self.Xlow ** (7 / 2)]
            c12 = [1, 0, 0, 0, 0, 0]
            clow = [c7, c8, c9, c10, c11, c12]

            return clow

        def BUp():
            bup = [
                (self.yTE + self.deltaYTE / 2),
                self.Yup,
                tan(-radians(self.alphaTE - self.betaTE / 2)),
                0,
                self.YXXup,
                sqrt(2 * self.rLE)
            ]
            return bup

        def BLow():
            blow = [
                (self.yTE - self.deltaYTE / 2),
                self.Ylow,
                tan(radians(self.alphaTE + self.betaTE / 2)),
                0,
                self.YXXlow,
                -(sqrt(2 * self.rLE))
            ]
            return blow

        self.AUp = tl.gaussJordonElimination(CUp(), BUp())
        self.ALow = tl.gaussJordonElimination(CLow(), BLow())

        # return AUp, ALow


def Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE):
    Airfoil = airfoil(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE)
    return Airfoil

