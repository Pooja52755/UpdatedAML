from flask import Flask, jsonify, request
import fraud_data
import decisions_db
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    df = fraud_data.get_transactions_df()
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/flagged_senders', methods=['GET'])
def get_flagged_senders():
    groups = fraud_data.get_all_flagged_senders()
    return jsonify(groups)

@app.route('/api/customer/<account_id>', methods=['GET'])
def get_customer(account_id):
    profile = fraud_data.get_customer_profile(account_id)
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify(profile)

@app.route('/api/decisions', methods=['GET'])
def get_decisions():
    decisions = decisions_db.get_analyst_decisions()
    return jsonify(decisions)

@app.route('/api/decision', methods=['POST'])
def post_decision():
    data = request.json
    if not data or not all(k in data for k in ("tx_id", "decision", "notes", "timestamp")):
        return jsonify({"error": "Missing required fields"}), 400
    
    decisions_db.record_analyst_decision(
        data["tx_id"], data["decision"], data["notes"], data["timestamp"]
    )
    return jsonify({"status": "success", "message": f"Recorded decision for {data['tx_id']}"})

if __name__ == '__main__':
    print("Starting AML Fraud API Backend...")
    app.run(port=5000, debug=True)
