from kafka import KafkaConsumer
import os
import json
import pandas as pd
from db import append_to_db
from cleaning import clean_new_row, append_to_csv

def start_consumer(kafka_url, topic_name):

    new_df = pd.DataFrame()

    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=kafka_url,
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print(f'Listening for messages in {topic_name}')

    done = False

    while not done:
        message = consumer.poll(timeout_ms=2000)
        print('Waiting for message')
        if message:
            for tp, messages in message.items():
                for msg in messages:
                    if(msg.value == 'EOF'):
                        print('End Of Stream')
                        done = True
                        break
                    new_row = pd.DataFrame([msg.value])
                    new_row = new_row.set_index(new_row.columns[0])

                    cleaned = clean_new_row(new_row)

                    new_df = pd.concat([new_df, cleaned])

                    append_to_db(cleaned)
                    append_to_csv(cleaned, './data/M2_cleaned_fintech_df.csv')
    
    new_df.to_csv('./data/cleaned.csv')
    consumer.close()

    print('Consumer DONE!')

