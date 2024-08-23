import json
import pandas as pd
import numpy as np
from nltk_utils import preprocess
from numpy.linalg import norm

# open and read the data from the intents.json
with open('intents.json', 'r', encoding="utf-8") as file:
    intents = json.load(file)

# preprocess the data
patterns = []
answers = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        answers.append(intent["responses"][0])



