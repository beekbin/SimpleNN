
from __future__ import print_function, division
import numpy as np

from activation import SigmoidActiveFunction
from activation import TanhActiveFunction


def vector_str(v):
    result = "["
    for e in v:
        result += ("%.3f  "%(e))

    result += "]"
    return result


def test_sigmoid():
    afunc = SigmoidActiveFunction()
    print(afunc)
    x = np.array([0, 1, -1, 2, -2, 4, -4])
    y = afunc.forward(x)
    dy = afunc.backward(y)
    print("x: %s" % vector_str(x))
    print("y: %s" % vector_str(y))
    print("dy: %s" % vector_str(dy))
    return


def test_tanh():
    afunc = TanhActiveFunction()
    print(afunc)
    x = np.array([0, 1, -1, 2, -2, 4, -4])
    y = afunc.forward(x)
    dy = afunc.backward(y)
    print("x: %s" % vector_str(x))
    print("y: %s" % vector_str(y))
    print("dy: %s" % vector_str(dy))
    return


def main():
    test_sigmoid()
    test_tanh()
    return


if __name__ == "__main__":
    main()
