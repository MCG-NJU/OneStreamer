import os, os.path as osp, json, math, re, string, warnings, logging, hashlib, pickle, csv
import numpy as np
import pandas as pd
import portalocker
from PIL import Image
from .file import *
from .log import get_logger
def listinstr(patterns, value): return any(p in value for p in patterns)
def istype(value, typ):
    try: typ(value); return True
    except (ValueError, TypeError): return False
