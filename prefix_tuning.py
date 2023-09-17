"""main hook to start the prefix tuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# code is heavily based on https://github.com/XiangLi1999/PrefixTuning and
# https://github.com/eth-sri/sven/tree/master

import os
import sys
import time
import datetime
import socket
import argparse
from typing import Final, List, Callable


def main() -> None:
    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefix Tuning")
    args = parser.parse_args()
    main(**vars(args))
