# To be able to import your function, you need to add the src/ directory to the Python path.
import pandas as pd
# For Label Encoding
from sklearn import preprocessing
from sqlalchemy import create_engine


from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def extract_clean(filename):
    df = pd.read_csv(filename)
    df = clean_missing(df)
    df.to_csv('/opt/airflow/data/titanic_clean.csv',index=False)
    print('loaded after cleaning succesfully')

def encode_load(filename):
    df = pd.read_csv(filename)
    df = encoding(df)
    try:
        df.to_csv('/opt/airflow/data/titanic_transformed.csv',index=False, mode='x')
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('file already exists')

def clean_missing(df):
    df = impute_mean(df,'Age')
    df = impute_arbitrary(df,'Cabin','Missing')
    df = cca(df,'Embarked')
    return df
def impute_arbitrary(df,col,arbitrary_value):
    df[col] = df[col].fillna(arbitrary_value)
    return df
def impute_mean(df,col):
    df[col] = df[col].fillna(df[col].mean())
    return df
def impute_median(df,col):
    df[col] = df[col].fillna(df[col].mean())
    return df
def cca(df,col):
    return df.dropna(subset=[col])
def encoding(df):
    df = one_hot_encoding(df,'Embarked')
    df = label_encoding(df,'Cabin')
    return df
def one_hot_encoding(df,col):
    to_encode = df[[col]]
    encoded = pd.get_dummies(to_encode)
    df = pd.concat([df,encoded],axis=1)
    return df
def label_encoding(df,col):
    df[col] = preprocessing.LabelEncoder().fit_transform(df[col])
    return df
def load_to_csv(df,filename):
    df.to_csv(filename,index=False)
    print('loaded succesfully')
    
def load_to_postgres(filename): 
    df = pd.read_csv(filename)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/titanic_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'titanic_passengers',con = engine,if_exists='replace')



# Define the DAG
default_args = {
    "owner": "data_engineering_team",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'titanic_etl_pipeline',
    default_args=default_args,
    description='titanic etl pipeline',
)

with DAG(
    dag_id = 'titanic_etl_pipeline',
    schedule_interval = '@once', # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args = default_args,
    tags = ['titanic-pipeline'],
)as dag:
    # Define the tasks
    extract_clean_task = PythonOperator(
        task_id = 'extract_clean',
        python_callable = extract_clean,
        op_kwargs = {
            'filename': '/opt/airflow/data/titanic.csv'
        }
    )

    encode_load_task = PythonOperator(
        task_id = 'encode_load',
        python_callable = encode_load,
        op_kwargs = {
            'filename': '/opt/airflow/data/titanic_clean.csv'
        }
    )

    load_to_postgres_task = PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs = {
            'filename': '/opt/airflow/data/titanic_transformed.csv'
        }
    )

    # Define the task dependencies
    extract_clean_task >> encode_load_task >> load_to_postgres_task