import os
import datetime
from psycopg2 import sql
import psycopg2.extras
from dotenv import load_dotenv
import yaml
from attrdict2 import AttrDict
import psycopg2
import pandas as pd

# Specify the path to config file
this_dir = os.path.dirname(__file__)
config_file = os.path.join(os.path.dirname(os.path.dirname(this_dir)), 'config.yaml')
# Open and read the YAML file
with open(config_file, 'r') as file:
    config = AttrDict(yaml.safe_load(file))

# Database credentials
load_dotenv()
DB_USER_NAME = os.getenv('DB_USER_NAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_URL = os.getenv('DB_URL')


def create_historical_data_table(db_url, db_user_name, db_password, table_name):
    db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
    cursor = db_connection.cursor()

    try:
        # Create the table if it doesn't exist
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                time TIMESTAMP,
                pm2_5 FLOAT,
                city VARCHAR(20),
                latitude FLOAT,
                longitude FLOAT,
                temperature_2m FLOAT,
                relativehumidity_2m INTEGER,
                precipitation FLOAT,
                cloudcover INTEGER,
                cloudcover_low INTEGER,
                cloudcover_mid INTEGER,
                cloudcover_high INTEGER,
                windspeed_10m FLOAT,
                winddirection_10m INTEGER,
                month INTEGER,
                hour INTEGER,
                PRIMARY KEY (time, city)
            )
        """).format(sql.Identifier(table_name))

        cursor.execute(create_table_query)
        db_connection.commit()

        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")
    return "Done"


def create_table(db_url, db_user_name, db_password, schema, table_name):
    db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
    cursor = db_connection.cursor()
    try:
        # Create the table if it doesn't exist
        create_table_query = sql.SQL(f"""
            CREATE TABLE IF NOT EXISTS {table_name} ({schema})
        """)

        cursor.execute(create_table_query)
        db_connection.commit()

        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")
    return "Done"


def create_inference_data_table(db_url, db_user_name, db_password, table_name='inference_data'):
    db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
    cursor = db_connection.cursor()

    try:
        # Create the table if it doesn't exist
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                time TIMESTAMP,
                city VARCHAR(20),
                latitude FLOAT,
                longitude FLOAT,
                temperature_2m FLOAT,
                relativehumidity_2m INTEGER,
                precipitation FLOAT,
                cloudcover INTEGER,
                cloudcover_low INTEGER,
                cloudcover_mid INTEGER,
                cloudcover_high INTEGER,
                windspeed_10m FLOAT,
                winddirection_10m INTEGER,
                month INTEGER,
                hour INTEGER,
                PRIMARY KEY (time, city)
            )
        """).format(sql.Identifier(table_name))

        cursor.execute(create_table_query)
        db_connection.commit()

        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")
    return "Done"


def create_predictions_data_table(db_url, db_user_name, db_password, table_name='predicted_data'):
    db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
    cursor = db_connection.cursor()

    try:
        # Create the table if it doesn't exist
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                time TIMESTAMP,
                city VARCHAR(20),
                pm2_5_pred FLOAT,
                PRIMARY KEY (time, city)
            )
        """).format(sql.Identifier(table_name))

        cursor.execute(create_table_query)
        db_connection.commit()

        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")
    return "Done"


def insert_df_to_table(db_url, db_user_name, db_password, table_name, df, expected_columns):
    try:
        db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
        cursor = db_connection.cursor()

        # Ensure the DataFrame has the expected columns and in this order
        df = df[expected_columns]

        # Insert data into the table
        column_string = ', '.join(expected_columns)
        insert_query = f"""
            INSERT INTO {table_name} (
            {column_string}
            )
            VALUES %s
            ON CONFLICT (time, city) DO NOTHING
        """
        # insert all the columns and rows in df
        # data_tuples = [tuple(x) for x in df.to_numpy()]
        # cursor.executemany(insert_query, data_tuples)

        # insert all the rows in df
        psycopg2.extras.execute_values(cursor, insert_query, df.values)

        db_connection.commit()
        print("Data loaded successfully.")

        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")


def delete_table(db_url, db_user_name, db_password, table_name):
    try:
        # Connect to the PostgreSQL database
        db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
        cursor = db_connection.cursor()

        # Drop the table if it exists
        drop_table_query = sql.SQL(f"""
            DROP TABLE IF EXISTS {table_name}
        """)

        cursor.execute(drop_table_query)
        db_connection.commit()

        print("Table deleted successfully.")
        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")


def close_idle_transactions(db_connection):
    try:
        # Identify the pid of the idle in transaction session
        cursor = db_connection.cursor()
        cursor.execute("""SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
                          WHERE state IN ('idle', 'idle in transaction');""")

        # Commit the transaction
        db_connection.commit()

        print("Idle in transaction sessions terminated.")

    except Exception as e:
        print("Error:", e)


def read_data_from_table(db_url, db_user_name, db_password, table_name, query=None):
    try:
        # Connect to the PostgreSQL database
        db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
        # Create a cursor
        cursor = db_connection.cursor()

        # Execute the query
        if query is None:
            query = f"SELECT * FROM {table_name}"
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Get column names from the cursor description
        column_names = [desc[0] for desc in cursor.description]

        # Convert to a DataFrame
        df = pd.DataFrame(rows, columns=column_names)

        print("Data loaded into DataFrame successfully.")

        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")

    return df


def delete_rows(db_url, db_user_name, db_password, table_name, n_days=60):

    try:
        # Connect to the PostgreSQL database
        db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
        # Create a cursor
        cursor = db_connection.cursor()

        threshold_date = datetime.datetime.now() - datetime.timedelta(days=n_days)
        # Define the DELETE query with LIMIT
        delete_query = f"DELETE FROM {table_name} WHERE time < '{threshold_date}'"

        # Execute the DELETE query with LIMIT using a parameterized query
        cursor.execute(delete_query)

        # Commit the transaction
        db_connection.commit()

        print(f"older than '{threshold_date}' entries are deleted.")
        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")


def run_vacuum(db_url, db_user_name, db_password):

    try:
        # Connect to the PostgreSQL database
        db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
        # Create a cursor
        cursor = db_connection.cursor()
        # Disable autocommit to run vacuum (not in a transaction block)
        db_connection.autocommit = True
        # optimize db with vacuum
        cursor.execute("VACUUM FULL VERBOSE")

        # Commit the transaction
        db_connection.commit()

        print(f"VACUUM run completed.")
        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Re-enable autocommit and close cursor and connection
        db_connection.autocommit = False  # Restore autocommit behavior
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")


def get_row_count(db_url, db_user_name, db_password, table_name):
    try:
        # Connect to the PostgreSQL database
        db_connection = psycopg2.connect(db_url, user=db_user_name, password=db_password)
        # Create a cursor
        cursor = db_connection.cursor()

        # Define the SELECT COUNT(*) query
        count_query = f"SELECT COUNT(*) FROM {table_name}"

        # Execute the query
        cursor.execute(count_query)

        # Fetch the result
        total_rows = cursor.fetchone()[0]

        close_idle_transactions(db_connection)

    except Exception as e:
        print("Error:", e)

    finally:
        # Close the connection
        if db_connection:
            cursor.close()
            db_connection.close()
            print("Connection closed.")

    return total_rows

