from cleaning import clean, read_from_csv, save_to_csv
import pandas as pd
from db import save_to_db
import os
from run_producer import start_producer, stop_container
from consumer import start_consumer

def main():

    # Removing existing csv (for testing purposes only)
    if (os.path.exists('./data/M2_lookup_table.csv')):
        os.remove('./data/M2_lookup_table.csv')
    # if (os.path.exists('./data/M2_cleaned_fintech_df.csv')):
    #     os.remove('./data/M2_cleaned_fintech_df.csv')
    if (os.path.exists('./data/addr_state_labels.csv')):
        os.remove('./data/addr_state_labels.csv')
    if (os.path.exists('./data/state_labels.csv')):
        os.remove('./data/state_labels.csv')
    if (os.path.exists('./data/letter_grade_labels.csv')):
        os.remove('./data/letter_grade_labels.csv')
    if (os.path.exists('./data/grade_int_rate.csv')):
        os.remove('./data/grade_int_rate.csv')
    if (os.path.exists('./data/emp_title_imputation.csv')):
        os.remove('./data/emp_title_imputation.csv')
    if (os.path.exists('./data/emp_length_imputation.csv')):
        os.remove('./data/emp_length_imputation.csv')
    if (os.path.exists('./data/grade_int_rate.csv')):
        os.remove('./data/grade_int_rate.csv')

        

    # Loading DF
    df = read_from_csv('./data/fintech_data_18_52_11870.csv')
    
    # Cleaning or loading if already cleaned
    if os.path.exists('./data/M2_lookup_table.csv') and os.path.exists('./data/M2_cleaned_fintech_df.csv'):
        cleaned = read_from_csv('./data/M2_cleaned_fintech_df.csv')
        lookup = read_from_csv('./data/M2_lookup_table.csv')
    else:
        cleaned, lookup = clean(df)
        save_to_csv(cleaned, './data/M2_cleaned_fintech_df.csv')
        save_to_csv(lookup, './data/M2_lookup_table.csv')


    # Saving to DB
    save_to_db(cleaned, lookup)

    # Run Producer
    id = start_producer('52_11870', 'kafka:9092', 'm2-topic')

    # Run Consumer
    start_consumer('kafka:9092', 'm2-topic')

    # Stop producer
    stop_container(id)




if __name__ == '__main__':
    main()