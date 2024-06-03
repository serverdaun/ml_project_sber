import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

with open('model/gbc_model.pkl', 'rb') as file:
    model = dill.load(file)

