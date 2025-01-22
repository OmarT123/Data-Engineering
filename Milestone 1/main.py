from cleaning import clean

def main():
    df = pd.read_csv('./Dataset/fintech_data_18_52_11870.csv', index_col=0)
    cleaned = clean(df)


if __name__ == "__main__":
    main()