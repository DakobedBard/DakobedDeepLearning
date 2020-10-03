import json
from datetime import datetime
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="postgres")

limit = 200
last_unix = 0
cur_length = limit
counter = 0
test_done = False

while cur_length == limit:

    df = pd.read_sql(
        # "SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format( last_unix, limit), conn)
        "SELECT * FROM parent_reply ", conn)
    last_unix = df.tail(1)['unix'].values[0]
    cur_length = len(df)

    if not test_done:
        with open('test.from', 'a', encoding='utf8') as f:
            for content in df['parent'].values:
                if not content:
                    continue
                f.write(content + '\n')

        with open('test.to', 'a', encoding='utf8') as f:
            for content in df['comment'].values:
                f.write(str(content) + '\n')

        test_done = True

    else:
        with open('train.from', 'a', encoding='utf8') as f:
            for content in df['parent'].values:
                f.write(content + '\n')

        with open('train.to', 'a', encoding='utf8') as f:
            for content in df['comment'].values:
                f.write(str(content) + '\n')

    counter += 1
    if counter % 20 == 0:
        print(counter * limit, 'rows completed so far')