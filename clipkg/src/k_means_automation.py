import numpy as np
import pandas as pd
import os
import sys

from datetime import datetime
from os import path
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline

from nlp_derog.src.util import json_handler
from nlp_derog import DATA_PATH, MODULE_PATH, MODEL_OUT_PATH
from nlp_derog.src.transformers import KMeansEstimator, StopWordExtractorTransformer, Doc2VecTransformer


# Notice each list is indexed at starting at 1, this is because the FIRST argument is ALWAYS the [file_name]
opts = [opt for opt in sys.argv[1:] if opt.startswith('-')]
args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

# if '-k' in opts:
#     print(' '.join(arg.capitalize() for arg in args))
# elif '-u' in opts:
#     print(' '.join(arg.upper() for arg in args))
# elif '-l' in opts:
#     print(' '.join(arg.lower() for arg in args))
# else:
#     raise SystemExit(f'Usage {sys.argv[0]} (-k | ) <arguments>')

# TODO: Only need args for now!

print(args)