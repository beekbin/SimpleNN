from __future__ import print_function

import sys
sys.path.insert(0, "../")
from util import mnist


def main():
    data = "../data/"

    train_data = mnist.load_data(data, "train")
    test_data = mnist.load_data(data, "test")

    return


if __name__ == "__main__":
    sys.exit(main())
