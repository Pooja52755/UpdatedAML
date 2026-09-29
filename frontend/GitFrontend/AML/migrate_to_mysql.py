import os
import glob
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

# TiDB connection details
DB_HOST = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
DB_PORT = 4000
DB_USER = "3JTMKeEP2m268Uk.root"
DB_PASSWORD = "KUOElkUyvNZX8HBL"
DB_NAME = "aml_fraud_db"

def find_file(pattern, directory):
    matches = glob.glob(os.path.join(directory, pattern))
    if matches:
        return matches[0]
    return None

def migrate_data():
    print("Connecting to MySQL Server...")
    # Connect to MySQL Server (without specifying a database) to create it if needed
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
        cursor.close()
        conn.close()
        print(f"Database '{DB_NAME}' ensured.")
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return

    # Use SQLAlchemy for pandas to_sql
    import urllib.parse
    encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        connect_args={"ssl": {"ssl_cert": None}} # PyMySQL basic SSL for TiDB
    )

    gitdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "GitData"))
    print(f"Looking for data in: {gitdata_dir}")

    # 1. Migrate Transactions
    trans_file = find_file("frontend (1).csv", gitdata_dir) or find_file("frontend*.csv", gitdata_dir)
    if trans_file:
        print(f"Migrating transactions from {trans_file}...")
        try:
            df_trans = pd.read_csv(trans_file)
            # Standardize column names (make them lowercase, replace spaces with underscores)
            df_trans.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_trans.columns]
            
            # Write to MySQL table
            df_trans.to_sql("transactions", con=engine, if_exists="replace", index=False)
            print("Successfully migrated 'transactions' table.")
        except Exception as e:
            print(f"Failed to migrate transactions: {e}")
    else:
        print("Transactions CSV not found!")

    # 2. Migrate Customer Profiles
    profiles_file = find_file("customer_profiles.csv", gitdata_dir) or find_file("*profile*.csv", gitdata_dir)
    if profiles_file:
        print(f"Migrating customer profiles from {profiles_file}...")
        try:
            df_prof = pd.read_csv(profiles_file)
            df_prof.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_prof.columns]
            
            df_prof.to_sql("customer_profiles", con=engine, if_exists="replace", index=False)
            print("Successfully migrated 'customer_profiles' table.")
        except Exception as e:
            print(f"Failed to migrate customer profiles: {e}")
    else:
        print("Customer Profiles CSV not found!")

    print("Migration complete!")

if __name__ == "__main__":
    migrate_data()

