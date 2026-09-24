"""
AML Fraud Detection Data & Engine Layer
Provides real transaction graph datasets from frontend_fanout_gat_test_grouped.csv,
real customer profiles, risk metrics, and fact-based pattern explanations.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

# Path to the real CSV dataset
DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "GitData",
        "frontend_fanout_gat_test_grouped.csv"
    )
)

CURRENCY_SYMBOLS = {
    "US Dollar": "$",
    "Euro": "€",
    "Rupee": "₹",
    "UK Pound": "£",
    "Yen": "¥",
    "Yuan": "¥",
    "Swiss Franc": "CHF ",
    "Canadian Dollar": "CA$",
    "Australian Dollar": "A$",
    "Ruble": "₽",
    "Shekel": "₪",
    "Bitcoin": "₿",
    "Mexican Peso": "Mex$",
    "Saudi Riyal": "SAR ",
    "Brazil Real": "R$",
}

def format_currency(amount, currency_name="US Dollar"):
    """Format numeric amount with proper currency symbol without forcing INR."""
    sym = CURRENCY_SYMBOLS.get(currency_name, f"{currency_name} ")
    try:
        amt = float(amount)
        if amt >= 1_000_000_000:
            return f"{sym}{amt / 1_000_000_000:.2f}B"
        elif amt >= 1_000_000:
            return f"{sym}{amt / 1_000_000:.2f}M"
        elif amt >= 1_000:
            return f"{sym}{amt:,.2f}"
        else:
            return f"{sym}{amt:.2f}"
    except Exception:
        return f"{sym}{amount}"


class DatasetManager:
    """Singleton manager that loads and indexes the real Fan-Out CSV data efficiently."""
    _instance = None
    _df = None
    _group_summaries = None
    _group_indices = None
    _all_senders = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        if self._df is not None:
            return

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset CSV not found at: {DATA_PATH}")

        # Read CSV with exact column names
        self._df = pd.read_csv(DATA_PATH)

        # Pre-build group index mapping for instant group lookups
        self._df["Fan-out Group"] = self._df["Fan-out Group"].astype(int)
        
        # Pre-calculate summary list of unique Fan-out groups (top 100 for fast left panel & navigation)
        unique_groups = self._df["Fan-out Group"].drop_duplicates().tolist()
        
        self._group_summaries = []
        for gid in unique_groups[:100]:
            gdf = self._df[self._df["Fan-out Group"] == gid]
            if gdf.empty:
                continue
            first_row = gdf.iloc[0]
            gat_prob = float(first_row["GAT Probability"])
            gat_signal = str(first_row["GAT Signal"]).strip()
            risk_tier = "High" if "HIGH" in gat_signal.upper() or gat_prob >= 0.7 else ("Medium" if gat_prob >= 0.3 else "Low")
            risk_score = int(round(gat_prob * 100))
            if risk_score > 100:
                risk_score = 100
            elif risk_score < 0:
                risk_score = 0

            currency = str(first_row["Payment Currency"])
            total_amt = gdf["Amount Paid"].sum()
            tx_id = str(first_row["Transaction ID"])
            from_acc = str(first_row["From Account"])
            from_entity = str(first_row["From Entity Name"])
            pattern = str(first_row["Detection Pattern"])
            timestamp = str(first_row["Timestamp"])
            payment_format = str(first_row["Payment Format"])
            actual_label = int(first_row["Actual Label"])
            prev_out = first_row["previous_outgoing"]
            prev_in = first_row["previous_incoming"]
            uniq_recv = gdf["To Account"].nunique()
            tx_count = len(gdf)

            summary = {
                "group_id": gid,
                "tx_id": f"GROUP-{gid}",
                "lead_tx_id": tx_id,
                "account": from_acc,
                "name": from_entity,
                "entity_id": str(first_row.get("From Entity ID", "—")),
                "bank_name": str(first_row.get("From Bank Name", "—")),
                "bank_id": str(first_row.get("From Bank ID", "—")),
                "risk": risk_tier,
                "risk_score": risk_score,
                "gat_prob": gat_prob,
                "gat_signal": gat_signal,
                "actual_label": actual_label,
                "pattern": pattern,
                "timestamp": timestamp,
                "payment_format": payment_format,
                "payment_currency": currency,
                "amount": total_amt,
                "amount_formatted": format_currency(total_amt, currency),
                "tx_count": tx_count,
                "unique_receivers": prev_out,
                "unique_senders": prev_in,
                "previous_outgoing": prev_out,
                "previous_incoming": prev_in,
                "is_fraud": (actual_label == 1),
                "model_used": "GAT AML Model",
                "model_confidence": f"{round(gat_prob * 100, 2)}%",
                "explanations": [
                    f"Fan-Out pattern detected: 1 sender ({from_acc}) → {uniq_recv} unique receivers",
                    f"{tx_count} transactions recorded in this fan-out group",
                    f"Previous outgoing transactions: {prev_out}",
                    f"Previous incoming transactions: {prev_in}",
                    f"GAT model probability: {gat_prob}",
                    f"GAT model signal: {gat_signal}",
                    f"Actual laundering ground truth: {actual_label} ({'Laundering' if actual_label == 1 else 'Normal'})",
                    f"Payment format: {payment_format} · Currency: {currency}",
                ]
            }
            if "Detection Reason" in first_row and pd.notna(first_row["Detection Reason"]):
                summary["explanations"].append(f"Detection reason: {first_row['Detection Reason']}")

            self._group_summaries.append(summary)

    def get_df(self):
        return self._df

    def get_group_summaries(self):
        return self._group_summaries

    def get_group_df(self, group_id):
        return self._df[self._df["Fan-out Group"] == int(group_id)]


# Initialize singleton accessor
_dm = DatasetManager.get_instance()

# Pre-populated TRANSACTIONS list representing Fan-out Groups
TRANSACTIONS = _dm.get_group_summaries()


def get_transactions_df():
    """Returns a pandas DataFrame of transactions from the dataset."""
    df = _dm.get_df()
    return df


def get_all_flagged_senders():
    """Returns unique flagged sender investigations (one per Fan-out Group)."""
    return _dm.get_group_summaries()


def get_flagged_transactions():
    """Returns all group summaries for navigation."""
    return _dm.get_group_summaries()


def get_transaction_by_id(tx_id_or_group_id):
    """Retrieve full details for a specific Fan-out Group or Transaction ID."""
    summaries = _dm.get_group_summaries()
    # Try exact match on tx_id or GROUP-X
    for s in summaries:
        if str(s.get("tx_id")) == str(tx_id_or_group_id) or str(s.get("group_id")) == str(tx_id_or_group_id) or str(s.get("lead_tx_id")) == str(tx_id_or_group_id):
            return s
    # Try parsing integer group_id
    try:
        gid = int(str(tx_id_or_group_id).replace("GROUP-", "").replace("TX-", ""))
        for s in summaries:
            if s.get("group_id") == gid:
                return s
        # If not in top summaries, load dynamically from df
        gdf = _dm.get_group_df(gid)
        if not gdf.empty:
            first_row = gdf.iloc[0]
            gat_prob = float(first_row["GAT Probability"])
            gat_signal = str(first_row["GAT Signal"]).strip()
            risk_tier = "High" if "HIGH" in gat_signal.upper() or gat_prob >= 0.7 else "Low"
            risk_score = int(round(gat_prob * 100))
            currency = str(first_row["Payment Currency"])
            total_amt = gdf["Amount Paid"].sum()
            uniq_recv = gdf["To Account"].nunique()
            tx_count = len(gdf)
            prev_out = first_row["previous_outgoing"]
            prev_in = first_row["previous_incoming"]
            return {
                "group_id": gid,
                "tx_id": f"GROUP-{gid}",
                "lead_tx_id": str(first_row["Transaction ID"]),
                "account": str(first_row["From Account"]),
                "name": str(first_row["From Entity Name"]),
                "entity_id": str(first_row.get("From Entity ID", "—")),
                "bank_name": str(first_row.get("From Bank Name", "—")),
                "bank_id": str(first_row.get("From Bank ID", "—")),
                "risk": risk_tier,
                "risk_score": risk_score,
                "gat_prob": gat_prob,
                "gat_signal": gat_signal,
                "actual_label": int(first_row["Actual Label"]),
                "pattern": str(first_row["Detection Pattern"]),
                "timestamp": str(first_row["Timestamp"]),
                "payment_format": str(first_row["Payment Format"]),
                "payment_currency": currency,
                "amount": total_amt,
                "amount_formatted": format_currency(total_amt, currency),
                "tx_count": tx_count,
                "unique_receivers": prev_out,
                "unique_senders": prev_in,
                "previous_outgoing": prev_out,
                "previous_incoming": prev_in,
                "is_fraud": (int(first_row["Actual Label"]) == 1),
                "model_used": "GAT AML Model",
                "model_confidence": f"{round(gat_prob * 100, 2)}%",
                "explanations": [
                    f"Fan-Out pattern detected: 1 sender ({first_row['From Account']}) → {uniq_recv} receivers",
                    f"{tx_count} transactions in this fan-out group",
                    f"Previous outgoing transactions: {prev_out}",
                    f"Previous incoming transactions: {prev_in}",
                    f"GAT probability: {gat_prob}",
                    f"GAT signal: {gat_signal}",
                    f"Actual laundering label: {first_row['Actual Label']}",
                ]
            }
    except Exception:
        pass

    return summaries[0] if summaries else {}


def get_fan_out_rows(tx_id_or_group_id):
    """Returns all sub-transaction rows for the selected Fan-out Group."""
    group_info = get_transaction_by_id(tx_id_or_group_id)
    gid = group_info.get("group_id", 1)
    gdf = _dm.get_group_df(gid)
    if gdf.empty:
        return []

    rows = []
    for _, r in gdf.iterrows():
        currency = str(r["Payment Currency"])
        amt_paid = float(r["Amount Paid"])
        amt_received = float(r["Amount Received"])
        rows.append({
            "sub_tx_id": str(r["Transaction ID"]),
            "to_account": str(r["To Account"]),
            "amount": format_currency(amt_paid, currency),
            "time": str(r["Timestamp"]),
            "raw_amount_paid": amt_paid,
            "raw_amount_received": amt_received,
            "currency": currency,
            "receiving_currency": str(r["Receiving Currency"]),
            "to_entity_name": str(r["To Entity Name"]),
            "to_entity_id": str(r["To Entity ID"]),
            "to_bank_name": str(r["To Bank Name"]),
            "to_bank_id": str(r["To Bank ID"]),
            "payment_format": str(r["Payment Format"]),
            "previous_outgoing": r["previous_outgoing"],
            "previous_incoming": r["previous_incoming"],
            "gat_prob": float(r["GAT Probability"]),
            "gat_signal": str(r["GAT Signal"]),
            "actual_label": int(r["Actual Label"]),
        })
    return rows


def get_customer_profile(account_id, *args, **kwargs):
    """Retrieve sender customer profile derived from actual CSV records."""
    if isinstance(account_id, dict):
        account_id = account_id.get("account", account_id.get("from_account", ""))

    df = _dm.get_df()
    sender_rows = df[df["From Account"].astype(str) == str(account_id)]
    if sender_rows.empty:
        # Check if account is in To Account
        recv_rows = df[df["To Account"].astype(str) == str(account_id)]
        if not recv_rows.empty:
            r = recv_rows.iloc[0]
            curr = str(r["Payment Currency"])
            tot_in = recv_rows["Amount Received"].sum()
            prev_out = r["previous_outgoing"]
            prev_in = r["previous_incoming"]
            return {
                "account_id": account_id,
                "name": str(r["To Entity Name"]),
                "entity_id": str(r["To Entity ID"]),
                "bank_name": str(r["To Bank Name"]),
                "bank_id": str(r["To Bank ID"]),
                "risk_tier": "High Risk" if "HIGH" in str(r["GAT Signal"]).upper() else "Low Risk",
                "total_transactions": str(len(recv_rows)),
                "total_incoming": format_currency(tot_in, curr),
                "unique_senders": prev_in,
                "unique_receivers": prev_out,
                "previous_incoming": prev_in,
                "previous_outgoing": prev_out,
                "avg_tx_amount": format_currency(recv_rows["Amount Received"].mean(), curr),
                "behavior_summary": [
                    {"metric": "Transactions", "historical": "—", "current": str(len(recv_rows)), "change": "—", "change_type": "low"},
                    {"metric": "Total Received", "historical": "—", "current": format_currency(tot_in, curr), "change": "—", "change_type": "low"},
                    {"metric": "Bank", "historical": "—", "current": str(r["To Bank Name"]), "change": "—", "change_type": "low"},
                ]
            }
        return {
            "account_id": account_id,
            "name": "—",
            "entity_id": "—",
            "bank_name": "—",
            "bank_id": "—",
            "risk_tier": "Unknown",
            "total_transactions": "—",
            "total_incoming": "—",
            "unique_senders": "—",
            "unique_receivers": "—",
            "avg_tx_amount": "—",
            "behavior_summary": []
        }

    first = sender_rows.iloc[0]
    currency = str(first["Payment Currency"])
    total_out = sender_rows["Amount Paid"].sum()
    tx_count = len(sender_rows)
    prev_out = first["previous_outgoing"]
    prev_in = first["previous_incoming"]
    gat_prob = float(first["GAT Probability"])
    gat_sig = str(first["GAT Signal"])

    return {
        "account_id": account_id,
        "name": str(first["From Entity Name"]),
        "entity_id": str(first.get("From Entity ID", "—")),
        "bank_name": str(first.get("From Bank Name", "—")),
        "bank_id": str(first.get("From Bank ID", "—")),
        "risk_tier": "High Risk" if "HIGH" in gat_sig.upper() or gat_prob >= 0.7 else "Low Risk",
        "total_transactions": str(tx_count),
        "total_incoming": "—",
        "total_outgoing": format_currency(total_out, currency),
        "unique_senders": prev_in,
        "unique_receivers": prev_out,
        "previous_outgoing": prev_out,
        "previous_incoming": prev_in,
        "avg_tx_amount": format_currency(sender_rows["Amount Paid"].mean(), currency),
        "behavior_summary": [
            {"metric": "Fan-Out Transactions", "historical": str(prev_out), "current": str(tx_count), "change": f"↑ {tx_count}", "change_type": "high"},
            {"metric": "Previous Inflows", "historical": str(prev_in), "current": "—", "change": "—", "change_type": "medium"},
            {"metric": "GAT Risk Score", "historical": "—", "current": f"{round(gat_prob*100)}/100", "change": gat_sig, "change_type": "high" if "HIGH" in gat_sig else "low"},
        ]
    }


def get_receiver_profile(account_id, group_id=None, *args, **kwargs):
    """Retrieve receiver profile from actual CSV dataset records on row click."""
    if isinstance(account_id, dict):
        d = account_id
        acc_str = str(d.get("to_account", d.get("account_id", "")))
        name = str(d.get("to_entity_name", d.get("name", "—")))
        bank_name = str(d.get("to_bank_name", d.get("bank_name", "—")))
        bank_id = str(d.get("to_bank_id", d.get("bank_id", "—")))
        entity_id = str(d.get("to_entity_id", d.get("entity_id", "—")))
        amt_str = str(d.get("amount", "—"))
        gat_sig = str(d.get("gat_signal", "HIGH RISK"))
        prev_in = d.get("previous_incoming", "—")
        prev_out = d.get("previous_outgoing", "—")
        return {
            "account_id": acc_str,
            "name": name,
            "entity_id": entity_id,
            "bank_name": bank_name,
            "bank_id": bank_id,
            "risk_tier": "High Risk" if "HIGH" in gat_sig.upper() else "Low Risk",
            "total_transactions": "1",
            "total_incoming": amt_str,
            "unique_senders": prev_in,
            "unique_receivers": prev_out,
            "previous_incoming": prev_in,
            "previous_outgoing": prev_out,
            "avg_tx_amount": amt_str,
            "notes": f"Entity: {name} ({entity_id}) | Bank: {bank_name} (ID: {bank_id})"
        }

    account_id = str(account_id)
    df = _dm.get_df()
    if group_id is not None:
        try:
            gid_int = int(str(group_id).replace("GROUP-", "").replace("TX-", ""))
            recv_rows = df[(df["Fan-out Group"] == gid_int) & (df["To Account"].astype(str) == account_id)]
        except Exception:
            recv_rows = df[df["To Account"].astype(str) == account_id]
    else:
        recv_rows = df[df["To Account"].astype(str) == account_id]

    if recv_rows.empty:
        recv_rows = df[df["To Account"].astype(str) == account_id]

    if recv_rows.empty:
        return {
            "account_id": account_id,
            "name": "—",
            "entity_id": "—",
            "bank_name": "—",
            "bank_id": "—",
            "risk_tier": "Unknown",
            "total_transactions": "—",
            "total_incoming": "—",
            "unique_senders": "—",
            "unique_receivers": "—",
            "avg_tx_amount": "—",
            "notes": "No profile data available"
        }

    first = recv_rows.iloc[0]
    curr = str(first["Payment Currency"])
    tot_in = recv_rows["Amount Received"].sum()
    avg_amt = recv_rows["Amount Received"].mean()
    tx_count = len(recv_rows)
    gat_sig = str(first["GAT Signal"])
    gat_prob = float(first["GAT Probability"])
    prev_in = first["previous_incoming"]
    prev_out = first["previous_outgoing"]

    to_entity_name = str(first["To Entity Name"])
    to_entity_id = str(first["To Entity ID"])
    to_bank_name = str(first["To Bank Name"])
    to_bank_id = str(first["To Bank ID"])

    return {
        "account_id": account_id,
        "name": to_entity_name,
        "entity_id": to_entity_id,
        "bank_name": to_bank_name,
        "bank_id": to_bank_id,
        "risk_tier": "High Risk" if "HIGH" in gat_sig.upper() or gat_prob >= 0.7 else "Low Risk",
        "total_transactions": str(tx_count),
        "total_incoming": format_currency(tot_in, curr),
        "unique_senders": prev_in,
        "unique_receivers": prev_out,
        "previous_incoming": prev_in,
        "previous_outgoing": prev_out,
        "avg_tx_amount": format_currency(avg_amt, curr),
        "notes": f"Entity: {to_entity_name} ({to_entity_id}) | Bank: {to_bank_name} (ID: {to_bank_id})"
    }


def create_network_graph(tx_id_or_group_id, include_2hop=False, max_nodes=20):
    """
    Generates a NetworkX directed graph for the selected Fan-out Group.
    Center = Sender (From Account)
    Hop-1 = Receivers (To Account) with actual transfer amounts.
    """
    group_info = get_transaction_by_id(tx_id_or_group_id)
    gid = group_info.get("group_id", 1)
    gdf = _dm.get_group_df(gid)

    G = nx.DiGraph()
    if gdf.empty:
        return G

    first_row = gdf.iloc[0]
    source_acc = str(first_row["From Account"])
    from_entity = str(first_row["From Entity Name"])
    from_bank = str(first_row["From Bank Name"])
    from_bank_id = str(first_row["From Bank ID"])
    from_entity_id = str(first_row["From Entity ID"])
    gat_prob = float(first_row["GAT Probability"])
    gat_signal = str(first_row["GAT Signal"])
    currency = str(first_row["Payment Currency"])

    # Sender Node (Star)
    G.add_node(
        source_acc,
        node_type="source",
        label=f"Sender\n{source_acc}",
        entity_name=from_entity,
        bank_name=from_bank,
        bank_id=from_bank_id,
        entity_id=from_entity_id,
        gat_prob=gat_prob,
        gat_signal=gat_signal,
        color="#ef4444" if "HIGH" in gat_signal.upper() else "#2563eb",
        hop=0
    )

    # Use first max_nodes receiver transactions to display exact transfer edges
    render_receivers = gdf.head(max_nodes)

    for _, r in render_receivers.iterrows():
        target = str(r["To Account"])
        to_entity = str(r["To Entity Name"])
        to_bank = str(r["To Bank Name"])
        to_bank_id = str(r["To Bank ID"])
        to_entity_id = str(r["To Entity ID"])
        amt = float(r["Amount Paid"])
        amt_str = format_currency(amt, currency)

        G.add_node(
            target,
            node_type="target",
            label=f"Receiver\n{target}",
            entity_name=to_entity,
            bank_name=to_bank,
            bank_id=to_bank_id,
            entity_id=to_entity_id,
            amount=amt_str,
            color="#f59e0b",
            hop=1
        )
        G.add_edge(source_acc, target, amount=amt_str, raw_amount=amt, currency=currency, hop=1)

    return G
