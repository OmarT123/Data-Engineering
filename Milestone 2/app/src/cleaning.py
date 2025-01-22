import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# import requests
# from bs4 import BeautifulSoup

def read_from_csv(path):
    df = pd.read_csv(path, index_col=0)
    print('Dataset loaded successfully!')
    return df

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

def copy_annual_inc_joint(df):
    df['annual_inc_joint_original'] = df['annual_inc_joint'].copy()

def fill_zero(df, col):
    df[col] = df[col].fillna(0)

def multivariate_median_imputation(df, group_col, missing_col):
    median_df = df.groupby(group_col)[missing_col].median().reset_index()
    median_df.columns = ['value', 'label']
    median_df.to_csv(f'./data/{group_col}_{missing_col}.csv', index=False)
    
    df[missing_col] = df.groupby(group_col)[missing_col].transform(lambda x: x.fillna(x.median()))
    

def impute_emp_title(df):
    bins = [0, 45_000, 80_000, 100_000, 200_000, 1e6 ,np.inf]
    labels = ['Low', 'Below Average', 'Average', 'Above Average', 'High', 'Extreme']

    df['income_category'] = pd.cut(df['annual_inc'], bins=bins, labels=labels)

    mode_df = []

    for category in df['income_category'].unique():
        mode_title = df[df['income_category'] == category]['emp_title'].mode()
        if not mode_title.empty:
            mode_df.append({'income_category': category, 'emp_title_mode': mode_title[0]})
            df.loc[(df['income_category'] == category) & (df['emp_title'].isna()), 'emp_title'] = mode_title[0]
    
    mode_df = pd.DataFrame(mode_df)
    mode_df.columns = ['value', 'label']
    mode_df.to_csv('./data/emp_title_imputation.csv', index=False)
    df = df.drop(columns=['income_category'])

def impute_emp_length(df):
    mode_emp_length = df.groupby('emp_title')['emp_length'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
  
    mode_df = mode_emp_length.reset_index()
    mode_df.columns = ['value', 'label']
    
    mode_df.to_csv('./data/emp_length_imputation.csv', index=False)

    df['emp_length'] = df.apply(lambda row: impute_emp_length_helper(row, mode_emp_length), axis=1)
    
def impute_emp_length_helper(row, mode_emp_length):
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
    new_encoded_df = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, new_encoded_df], axis=1)
    df = df.drop(column_name, axis=1)
    return df

def one_hot_encoding_row(df, column_name):
    new_encoded_df = pd.get_dummies(df[column_name], prefix=column_name)
    
    for col in new_encoded_df.columns:
        df[col] = True

    # df = pd.concat([df, new_encoded_df], axis=1)

    df = df.drop(column_name, axis=1)
    return df

def label_encoding(df, column_name, label_column_name):
    encoder = LabelEncoder()
    df[label_column_name] = encoder.fit_transform(df[column_name])

    mapping_df = pd.DataFrame({
        'value': encoder.classes_,
        'label': range(len(encoder.classes_))
    })
    
    mapping_df.to_csv(f'./data/{column_name}_labels.csv', index=False)
    print(f'Mapping saved to {column_name}_labels.csv')

def create_lookup_table(df, original_df):
    mapping = {
    'original': ['emp_length', 'int_rate', 'grade', 'zip_code', 'addr_state', 'annual_inc', 'avg_cur_bal', 'tot_cur_bal', 'total_annual_inc', 'loan_amount', 'funded_amount', 
    'installement_per_month', 'state'],
    'new_name': ['emp_length_years', 'sqrt_int_rate', 'letter_grade', 'zip_code_header', 'addr_state_label', 'normalized_annual_inc', 'normalized_avg_cur_bal', 'normalized_tot_cur_bal'
    ,'normalized_total_annual_inc', 'normalized_loan_amount', 'normalized_funded_amount', 'normalized_installement_per_month', 'state_name']
    }
    lookup_table = create_lookup_table_helper(original_df, df).drop_duplicates()
    features_to_keep = ['emp_length', 'addr_state', 'state', 'grade']
    cleaned_lookup_table = lookup_table[lookup_table['Feature'].isin(features_to_keep)]
    data = {
    'Feature': ['annual_inc_joint', 'home_ownership'],
    'Original': ['Nan', 'OTHER'],
    'Imputed': ['0', 'ANY']
    }
    cleaned_lookup_table = pd.concat([cleaned_lookup_table, pd.DataFrame(data)], ignore_index=True)
    
    return cleaned_lookup_table

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

# def fetch_state_data(response):
#     soup = BeautifulSoup(response.content, 'html.parser')
#     table = soup.find('table')
    
#     if not table:
#         raise ValueError("No table found in the HTML response.")

#     rows = table.find_all('tr')
#     state_codes = []
#     state_names = []
    
#     for row in rows[1:]:
#         columns = row.find_all('td')
#         if len(columns) == 3:
#             state_code = columns[2].text.strip()
#             state_name = columns[0].text.strip()
#             state_codes.append(state_code)
#             state_names.append(state_name)

#     return pd.DataFrame({'state_code': state_codes, 'state_name': state_names})

# def map_state_names(final_df, state_df):
#     state_dict = dict(zip(state_df['state_code'], state_df['state_name']))
#     final_df['state_name'] = final_df['state'].map(state_dict)

# def get_state_names(response, df):
#     if response.status_code == 200:
#         state_df = fetch_state_data(response)
#         print(state_df.head())
        
#         map_state_names(df, state_df)
#     else:
#         print(f"Failed to retrieve data. Status code: {response.status_code}")

def clean(fintech_df):
    print('Started Cleaning')

    # Convert column names to lower case and remove spaces
    rename_columns(fintech_df)

    # Keep original copy of df
    fintect_df_original = fintech_df.copy()

    print('Fixing Inconsistent Data')
    # Replace Inconsistent values
    replace_value(fintech_df, 'type', 'Individual', 'INDIVIDUAL')
    replace_value(fintech_df, 'type', 'Joint App', 'JOINT')
    replace_value(fintech_df, 'home_ownership', 'OTHER', 'ANY')
    handle_emp_tile_inconsistency(fintech_df)
    replace_value(fintech_df, 'emp_title', 'maintenace', 'maintenance')
    replace_value(fintech_df, 'emp_title' ,'sales and man', 'sales and management')

    # Drop description column
    drop_column(fintech_df, 'description')

    # Fixing dtypes
    change_to_datetime(fintech_df, 'issue_date')

    # Adding cols
    copy_annual_inc_joint(fintech_df)

    print('Handling Missing Values')
    # Handling missing values
    fill_zero(fintech_df, 'annual_inc_joint')
    multivariate_median_imputation(fintech_df, 'grade', 'int_rate') # FE MO4KELA HENA!!
    impute_emp_title(fintech_df)
    impute_emp_length(fintech_df)

    # Fixing dtypes
    convert_emp_length(fintech_df)

    print('Handling Outliers')
    # Handling outliers
    log_transform_outliers(fintech_df, 'annual_inc')
    log_transform_outliers(fintech_df, 'tot_cur_bal')
    log_transform_outliers(fintech_df, 'avg_cur_bal')
    log_transform_outliers(fintech_df, 'annual_inc_joint_original')

    # Adding new columns
    create_month_number_col(fintech_df)
    create_salary_can_cover_col(fintech_df)
    create_letter_grade_col(fintech_df)
    create_installement_per_month_col(fintech_df)

    # Handling outliers
    log_transform_outliers(fintech_df, 'installement_per_month')

    # Encoding
    change_zip_code_to_int(fintech_df)

    # Dropping unnecessary columns
    fintech_df = fintech_df.drop(['emp_length', 'annual_inc_joint', 'zip_code'], axis=1)

    print('Encoding')
    # Encoding
    fintech_df = one_hot_encoding(fintech_df, 'home_ownership')
    fintech_df = one_hot_encoding(fintech_df, 'verification_status')
    fintech_df = one_hot_encoding(fintech_df, 'loan_status')
    fintech_df = one_hot_encoding(fintech_df, 'term')
    fintech_df = one_hot_encoding(fintech_df, 'type')
    label_encoding(fintech_df, 'letter_grade', 'grade_label')
    label_encoding(fintech_df, 'state', 'state_label')
    label_encoding(fintech_df, 'addr_state', 'addr_state_label')

    print('Normalization')
    # Normalization
    log_transform_outliers(fintech_df, 'loan_amount')
    log_transform_outliers(fintech_df, 'funded_amount')

    print('Create Lookup Table')
    # LOOKUP TABLE
    lookup = create_lookup_table(fintech_df, fintect_df_original)
    
    # BONUS
    # url = "https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971"
    # response = requests.get(url)
    # get_state_names(response, fintech_df)

    return fintech_df, lookup

def get_label_from_csv(value, csv_path):
    mapping_df = pd.read_csv(csv_path)
    result = mapping_df.loc[mapping_df['value'] == value, 'label']

    if not result.empty:
        return result.iloc[0]
    else:
        print("Value not found in the label mapping.")
        return None

def get_income_category(annual_inc):
    return 'LOW'

def clean_new_row(row):
    print('Started Cleaning')

    # Convert column names to lower case and remove spaces
    rename_columns(row)

    print('Fixing Inconsistent Data')
    # Replace Inconsistent values
    replace_value(row, 'type', 'Individual', 'INDIVIDUAL')
    replace_value(row, 'type', 'Joint App', 'JOINT')
    replace_value(row, 'home_ownership', 'OTHER', 'ANY')
    handle_emp_tile_inconsistency(row)
    replace_value(row, 'emp_title', 'maintenace', 'maintenance')
    replace_value(row, 'emp_title' ,'sales and man', 'sales and management')

    # Drop description column
    drop_column(row, 'description')

    # Fixing dtypes
    change_to_datetime(row, 'issue_date')

    # Adding cols
    copy_annual_inc_joint(row)

    print('Handling Missing Values')
    # Handling missing values
    fill_zero(row, 'annual_inc_joint')
    if pd.isna(row['int_rate'].iloc[0]):
        row['int_rate'] = get_label_from_csv(row['int_rate'].iloc[0], './data/grade_int_rate.csv')
    if pd.isna(row['emp_title'].iloc[0]):    
        row['emp_title'] = get_label_from_csv(get_income_category(row['annual_inc'].iloc[0]), './data/emp_title_imputation.csv')
    if pd.isna(row['emp_length'].iloc[0]):    
        row['emp_length'] = get_label_from_csv(row['emp_title'].iloc[0], './data/emp_length_imputation.csv')

    # Fixing dtypes
    convert_emp_length(row)

    print('Handling Outliers')
    # Handling outliers
    log_transform_outliers(row, 'annual_inc')
    log_transform_outliers(row, 'tot_cur_bal')
    log_transform_outliers(row, 'avg_cur_bal')
    log_transform_outliers(row, 'annual_inc_joint_original')

    # Adding new columns
    create_month_number_col(row)
    create_salary_can_cover_col(row)
    create_letter_grade_col(row)
    create_installement_per_month_col(row)

    # Handling outliers
    log_transform_outliers(row, 'installement_per_month')

    # Encoding
    change_zip_code_to_int(row)

    # Dropping unnecessary columns
    row = row.drop(['emp_length', 'annual_inc_joint', 'zip_code'], axis=1)

    # Add all one_hot_encodindg columns
    additional_headers = pd.read_csv("./data/one_hot_encoded_cols.csv", header=None).iloc[0].str.strip().tolist()
    for header in additional_headers:
        row[header] = False

    print('Encoding')
    # Encoding
    row = one_hot_encoding_row(row, 'home_ownership')
    row = one_hot_encoding_row(row, 'verification_status')
    row = one_hot_encoding_row(row, 'loan_status')
    row = one_hot_encoding_row(row, 'term')
    row = one_hot_encoding_row(row, 'type')
    
    # Encoding from saved lookup tables
    row['grade_label'] = get_label_from_csv(row['letter_grade'].iloc[0], './data/letter_grade_labels.csv')
    row['state_label'] = get_label_from_csv(row['state'].iloc[0], './data/state_labels.csv')
    row['addr_state_label'] = get_label_from_csv(row['addr_state'].iloc[0], './data/addr_state_labels.csv')

    print('Normalization')
    # Normalization
    log_transform_outliers(row, 'loan_amount')
    log_transform_outliers(row, 'funded_amount')

    # BONUS


    return row

def save_to_csv(df, name):
    print('Saving DF to CSV')
    df.to_csv(name)
    print('DF Saved to CSV')

def append_to_csv(df, name):
    print('Appending DF to CSV')
    df.to_csv(name, mode='a', header=False)
    print('DF Added to CSV')