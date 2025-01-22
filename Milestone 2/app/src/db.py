from sqlalchemy import create_engine

engine = create_engine('postgresql://root:root@pgdatabase:5432/testdb')

def save_to_db(cleaned, lookup):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            cleaned.to_sql('fintech_db_MET_P01_52_11870', con=engine, if_exists='replace')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
        try:
            print('Writing lookup table to database')
            lookup.to_sql('lookup_fintech_data_MET_P01_52_11870S', con=engine, if_exists='replace')
            print('Done writing to database')
        except ValueError as vx:
            print('Lookup Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')


def append_to_db(cleaned):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            cleaned.to_sql('fintech_db_MET_P01_52_11870', con=engine, if_exists='append')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')