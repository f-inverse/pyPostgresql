import psycopg2
import pandas as pd
import numpy as np
from config import config

params = config()
conn = psycopg2.connect(**params)
cur = conn.cursor()

cur.close()
conn.close()


docker build -t postgres_client:v0 -f .devcontainer/Dockerfile . 
docker build -t icirauqui/postgres_client:v0 -f ./devcontainer/Dockerfile .