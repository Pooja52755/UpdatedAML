
# --- Analyst Decisions Database ---

def record_analyst_decision(tx_id, decision, notes, timestamp):
    """
    Saves an auditor decision to the MySQL database.
    """
    import mysql.connector
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Aryan@#1612",
            database="aml_fraud_db"
        )
        cursor = conn.cursor()
        
        sql = """
        INSERT INTO analyst_decisions (tx_id, decision, notes, timestamp)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            decision = VALUES(decision), 
            notes = VALUES(notes), 
            timestamp = VALUES(timestamp)
        """
        cursor.execute(sql, (tx_id, decision, notes, timestamp))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Recorded decision for {tx_id} in MySQL.")
    except Exception as e:
        print(f"Error saving decision to MySQL: {e}")

def get_analyst_decisions():
    """
    Retrieves all decisions from the MySQL database.
    """
    import mysql.connector
    decisions = {}
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Aryan@#1612",
            database="aml_fraud_db"
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM analyst_decisions")
        for row in cursor.fetchall():
            decisions[row['tx_id']] = {
                "decision": row['decision'],
                "notes": row['notes'],
                "timestamp": row['timestamp']
            }
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error retrieving decisions from MySQL: {e}")
    return decisions
