#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np

dataset, classcolumn, headers = sys.argv[1:]
headers = None if headers == "None" else headers
classcolumn = int(classcolumn) if headers == None else classcolumn

data = pd.read_csv(dataset, header=headers)
data.head()
print(classcolumn)