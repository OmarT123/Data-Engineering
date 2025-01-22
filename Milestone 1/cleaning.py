import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import requests
from bs4 import BeautifulSoup

plt.style.use('ggplot')
pd.set_option("display.max_columns", None)


def rename_columns(df):
    df.columns = df.columns.str.lower()
    df.columns = [col.replace(' ', '_') for col in df.columns]

def replace_value(df, column_name, old_value, new_value):
    df[column_name] = df[column_name].replace({old_value:new_value})

def handle_emp_tile_inconsistency(df):
    df['emp_title'] = df['emp_title'].str.lower().str.strip()

def drop_column(df, col):
    df = df.drop(col, axis=1)

def change_to_datetime(df, col):
    df[col] = pd.to_datetime(df[col])

def fill_zero(df, col):
    df[col] = df[col].fillna(0)

def multivariate_median_imputation(df, group_col, missing_col):
    df[missing_col] = df.groupby(group_col)[missing_col].transform(lambda x: x.fillna(x.median()))

def impute_emp_title(df):
    bins = [0, 45_000, 80_000, 100_000, 200_000, 1e6 ,np.inf]
    labels = ['Low', 'Below Average', 'Average', 'Above Average', 'High', 'Extreme']

    df['income_category'] = pd.cut(df['annual_inc'], bins=bins, labels=labels)

    for category in df['income_category'].unique():
        mode_title = df[df['income_category'] == category]['emp_title'].mode()
        if not mode_title.empty:
            df.loc[(df['income_category'] == category) & (df['emp_title'].isna()), 'emp_title'] = mode_title[0]
    df = df.drop(columns=['income_category'])

def impute_emp_length(df):
    mode_emp_length = consistent_df.groupby('emp_title')['emp_length'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    df['emp_length'] = df.apply(impute_emp_length_helper, axis=1)

def impute_emp_length_helper(row):
    if pd.isna(row['emp_length']) and pd.notna(row['emp_title']):
        return mode_emp_length[row['emp_title']]
    return row['emp_length']

def convert_emp_length(df):
    df['emp_length_years'] = df['emp_length'].apply(convert_emp_length_helper)

def convert_emp_length_helper(emp_length):
    if pd.isnull(emp_length):
        return np.nan
    elif emp_length == '10+ years':
        return 10
    elif emp_length == '< 1 year':
        return 0
    else:
        return int(emp_length.split()[0])

def remove_outliers_cap(df, column, new_column_name, lower_threshold, upper_threshold):
    df[new_column_name] = np.where(df[column] > upper_threshold, upper_threshold, df[column])
    df[new_column_name] = np.where(df[new_column_name] < lower_threshold, lower_threshold, df[new_column_name])

def log_transform_outliers(df, column_name):
    df[column_name] = np.log(df[column_name] + 1)

def create_month_number_col(df):
    df['month_number'] = df['issue_date'].dt.month

def create_salary_can_cover_col(df):
    inverse_log_annual_inc = np.exp(df['annual_inc'])
    inverse_log_loan_amount = np.exp(df['loan_amount'])
    df['salary_can_cover'] = inverse_log_annual_inc >= inverse_log_loan_amount

def map_grade(grade):
    if 1 <= grade <= 5:
        return 'A'
    elif 6 <= grade <= 10:
        return 'B'
    elif 11 <= grade <= 15:
        return 'C'
    elif 16 <= grade <= 20:
        return 'D'
    elif 21 <= grade <= 25:
        return 'E'
    elif 26 <= grade <= 30:
        return 'F'
    elif 31 <= grade <= 35:
        return 'G'
    else:
        return 'No Grade'

def create_letter_grade_col(df):
    df['letter_grade'] = df['grade'].apply(map_grade).astype('category')

def calculate_installement(P, r, n):
    return P * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def create_installement_per_month_col(df):
    df['monthly_rate'] = df['int_rate'] / 12
    df['term_int'] = df['term'].str.extract('(\d+)').astype(int)
    df['installement_per_month'] = df.apply(lambda row: calculate_installement(row['loan_amount'], row['monthly_rate'], row['term_int']), axis=1)
    df = df.drop('term_int', axis=1)

def convert_zip_code(zip_code):
    if pd.isnull(zip_code):
        return np.nan
    else:
        return int(zip_code[:3])

def change_zip_code_to_int(df):
    df['zip_code_header'] = df['zip_code'].apply(convert_zip_code)

def one_hot_encoding(df, column_name):
    new_encoded_df = pd.get_dummies(encoding_df[column_name], prefix=column_name)
    df = pd.concat([df, new_encoded_df], axis=1)
    df = df.drop(column_name, axis=1)

def label_encoding(df, column_name, label_column_name):
    encoder = LabelEncoder()
    df[label_column_name] = encoder.fit_transform(df[column_name])

def create_lookup_table(df, original_df):
    mapping = {
    'original': ['emp_length', 'int_rate', 'grade', 'zip_code', 'addr_state', 'annual_inc', 'avg_cur_bal', 'tot_cur_bal', 'total_annual_inc', 'loan_amount', 'funded_amount', 
    'installement_per_month', 'state'],
    'new_name': ['emp_length_years', 'sqrt_int_rate', 'letter_grade', 'zip_code_header', 'addr_state_label', 'normalized_annual_inc', 'normalized_avg_cur_bal', 'normalized_tot_cur_bal'
    ,'normalized_total_annual_inc', 'normalized_loan_amount', 'normalized_funded_amount', 'normalized_installement_per_month', 'state_name']
    }
    lookup_table = create_lookup_table_helper(fintech_df, final_df).drop_duplicates()
    features_to_keep = ['emp_length', 'addr_state', 'state', 'grade']
    cleaned_lookup_table = lookup_table[lookup_table['Feature'].isin(features_to_keep)]
    data = {
    'Feature': ['annual_inc_joint', 'home_ownership'],
    'Original': ['Nan', 'OTHER'],
    'Imputed': ['0', 'ANY']
    }
    cleaned_lookup_table = pd.concat([cleaned_lookup_table, pd.DataFrame(data)], ignore_index=True)
    
    cleaned_lookup_table.to_csv('M2_lookup_table.csv', index=False)


def create_lookup_table_helper(original_df, cleaned_df):
    lookup_list = []

    for original_col in original_df.columns:
        for cleaned_col in cleaned_df.columns:
            if original_col in cleaned_col:

                original_values = original_df[original_col].fillna('missing').astype(str)
                cleaned_values = cleaned_df[cleaned_col].astype(str)
                
                differences = original_values != cleaned_values
                    
                changed_indices = differences[differences].index
                lookup_list.extend([{
                    'Feature': original_col,
                    'Original': original_values.loc[index],
                    'Imputed': cleaned_values.loc[index]
                } for index in changed_indices])
                lookup_table = pd.DataFrame(lookup_list)
    return lookup_table

def clean(fintech_df):
    fintect_df_original = fintech_df.copy()

    # Convert column names to lower case and remove spaces
    rename_columns(fintech_df)

    # Replace Inconsistent values
    replace_value(fintech_df, 'type', 'Individual', 'INDIVIDUAL')
    replace_value(fintech_df, 'type', 'Joint App', 'JOINT')
    replace_value(fintech_df, 'home_ownership', 'OTHER', 'ANY')
    handle_emp_tile_inconsistency()
    replace_value(fintech_df, 'emp_title', 'maintenace', 'maintenance')
    replace_value(fintech_df, 'emp_title' ,'sales and man', 'sales and management')

    # Drop description column
    drop_column(fintech_df, 'description')

    # Fixing dtypes
    change_to_datetime(fintech_df, 'issue_date')

    # Handling missing values
    fill_zero(fintech_df, 'annual_inc_joint')
    multivariate_median_imputation(fintech_df, 'grade', 'int_rate') # FE MO4KELA HENA!!
    impute_emp_title(df)
    impute_emp_length(df)

    # Fixing dtypes
    convert_emp_length(fintech_df)

    # Handling outliers
    apply_log_transformation(fintech_df, 'annual_inc')
    apply_log_transformation(fintech_df, 'tot_cur_bal')
    apply_log_transformation(fintech_df, 'avg_cur_bal')
    apply_log_transformation(fintech_df, 'annual_inc_joint_original')

    # Adding new columns
    create_month_number_col(fintech_df)
    create_salary_can_cover_col(fintech_df)
    create_letter_grade_col(fintech_df)
    create_installement_per_month_col(fintech_df)

    # Handling outliers
    apply_log_transformation(fintech_df, 'installement_per_month')

    # Encoding
    change_zip_code_to_int(fintech_df)

    # Dropping unnecessary columns
    fintech_df = fintech_df.drop(['emp_length', 'annual_inc_joint', 'zip_code'], axis=1)

    # Encoding
    one_hot_encoding(fintech_df, 'home_ownership')
    one_hot_encoding(fintech_df, 'verification_status')
    one_hot_encoding(fintech_df, 'loan_status')
    one_hot_encoding(fintech_df, 'term')
    one_hot_encoding(fintech_df, 'type')
    label_encoding(fintech_df, 'letter_grade', 'grade_label')
    label_encoding(fintech_df, 'state', 'state_label')
    label_encoding(fintech_df, 'addr_state', 'addr_state_label')

    # Normalization
    apply_log_transformation(fintech_df, 'loan_amount')
    apply_log_transformation(fintech_df, 'funded_amount')

    # LOOKUP TABLE
    create_lookup_table(fintech_df, fintect_df_original)
    # BONUS

    # Saving to CSV
    fintech_df.to_csv('M2_cleaned_fintech_df.csv')
    
    return fintech_df