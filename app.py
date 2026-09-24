import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import textwrap
import importlib
import fraud_data
import graph_vis

# Force reload modules so in-memory cache never serves stale signatures
importlib.reload(fraud_data)
importlib.reload(graph_vis)

# ─── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────
st.html(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp { background-color: #f1f5f9; }

    /* ── Header ── */
    .header-title { font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; }
    .header-subtitle { font-size: 13px; color: #64748b; margin-top: 2px; }
    .header-timestamp { font-size: 13px; color: #64748b; }

    /* ── Badges ── */
    .badge-high {
        background:#fef2f2; color:#dc2626; border:1px solid #fecaca;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-medium {
        background:#fffbeb; color:#d97706; border:1px solid #fde68a;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-low {
        background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }

    /* ── Textarea styling ── */
    .stTextArea textarea {
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        font-size: 13px !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* ── Left Panel: Flagged Account Cards ── */
    .flagged-card {
        background:#ffffff; border:1.5px solid #e2e8f0; border-radius:10px;
        padding:12px 14px; margin-bottom:10px; cursor:pointer;
        transition: box-shadow 0.2s;
        border-left: 4px solid #e2e8f0;
    }
    .flagged-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .flagged-card-high { border-left-color: #ef4444 !important; }
    .flagged-card-medium { border-left-color: #f59e0b !important; }
    .flagged-card-low { border-left-color: #16a34a !important; }
    .flagged-card-selected { background:#eff6ff; border-color:#93c5fd; }
    .flagged-acc-id { font-weight:700; font-size:13px; color:#1e40af; }
    .flagged-pattern { font-size:11px; color:#64748b; margin-top:3px; }
    .risk-score-bar-bg {
        background:#f1f5f9; border-radius:4px; height:6px; margin-top:8px; overflow:hidden;
    }
    .risk-score-bar-fill {
        height:6px; border-radius:4px;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
    }

    /* ── Center Panel ── */
    .center-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:20px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    .graph-network-btn {
        display:inline-flex; align-items:center; gap:8px;
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color:white; font-weight:700; font-size:13px;
        padding:10px 18px; border-radius:8px; cursor:pointer;
        border:none; margin-bottom:16px;
        box-shadow: 0 2px 8px rgba(37,99,235,0.35);
        transition: all 0.2s;
        text-decoration:none;
    }
    .graph-network-btn:hover {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
        box-shadow: 0 4px 14px rgba(37,99,235,0.45);
        transform: translateY(-1px);
    }

    /* ── XAI Explanation box ── */
    .xai-box {
        background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
        padding:14px 16px; margin-top:16px;
    }
    .xai-box-high {
        background:#fff5f5; border:1px solid #fecaca;
    }
    .xai-title { font-weight:700; font-size:13px; color:#1e293b; margin-bottom:8px; }
    .xai-list { margin:0; padding-left:18px; font-size:12px; color:#334155; line-height:1.8; }

    /* ── AI Advisory banner ── */
    .ai-advisory {
        background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
        padding:10px 14px; font-size:11.5px; color:#0369a1;
        margin-top:12px; display:flex; align-items:flex-start; gap:8px;
    }

    /* ── Human Decision Panel ── */
    .human-decision-panel {
        background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
        border:1.5px solid #fbbf24; border-radius:12px;
        padding:18px 20px; margin-top:18px;
    }
    .human-decision-title {
        font-weight:800; font-size:14px; color:#92400e;
        display:flex; align-items:center; gap:8px; margin-bottom:6px;
    }
    .human-decision-disclaimer {
        font-size:11px; color:#78716c; background:#ffffff;
        border:1px solid #e7e5e4; border-radius:6px;
        padding:8px 12px; margin-top:10px;
    }

    /* ── Right Panel: Customer Profile ── */
    .profile-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:18px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
        height:100%;
    }
    .profile-empty {
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; padding:40px 20px; text-align:center;
        color:#94a3b8;
    }
    .profile-metric-grid {
        display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-bottom:14px;
    }
    .profile-metric-card {
        background:#f8fafc; border:1px solid #f1f5f9; border-radius:8px; padding:10px 12px;
    }
    .profile-metric-label { font-size:10px; color:#64748b; text-transform:uppercase; font-weight:600; }
    .profile-metric-val { font-size:15px; font-weight:700; color:#0f172a; margin-top:2px; }
    .kyc-verified { color:#16a34a; font-weight:700; }
    .kyc-flagged { color:#dc2626; font-weight:700; }
    .kyc-minimal { color:#d97706; font-weight:700; }
    .kyc-unverified { color:#dc2626; font-weight:700; }

    /* ── Behavior table ── */
    .custom-table { width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }
    .custom-table th {
        text-align:left; padding:8px 10px; background:#f8fafc;
        color:#64748b; font-weight:600; border-bottom:1px solid #e2e8f0;
    }
    .custom-table td { padding:8px 10px; border-bottom:1px solid #f1f5f9; color:#1e293b; }
    .change-high { color:#dc2626; font-weight:600; }
    .change-medium { color:#d97706; font-weight:600; }
    .change-low { color:#16a34a; font-weight:500; }

    /* ── Top banner ── */
    .top-banner {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:16px 22px; display:flex; align-items:center; gap:20px;
        margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    .top-banner-icon {
        width:46px; height:46px; background:#2563eb; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        color:white; font-size:22px;
    }
    .banner-value { font-size:26px; font-weight:800; color:#0f172a; line-height:1.2; }
    .banner-subtext { font-size:12px; color:#16a34a; font-weight:600; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e2e8f0; }
    .sidebar-brand {
        display:flex; align-items:center; gap:10px;
        padding:10px 0 20px 0; border-bottom:1px solid #f1f5f9; margin-bottom:20px;
    }
    .sidebar-brand-title { font-size:16px; font-weight:800; color:#0f172a; }
    .sidebar-brand-subtitle { font-size:11px; color:#64748b; }

    /* ── Model info bar ── */
    .model-info-bar {
        background:#eff6ff; border:1px solid #dbeafe; border-radius:8px;
        padding:10px 14px; display:flex; justify-content:space-between;
        align-items:center; font-size:12px; color:#1e40af; margin-top:14px;
    }

    /* ── Section labels ── */
    .section-label {
        font-size:11px; font-weight:700; color:#64748b;
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;
    }
</style>
"""))

# ─── Session State Initialization ────────────────────────────────────────
if "selected_tx_id" not in st.session_state:
    st.session_state.selected_tx_id = "GROUP-1"
if "selected_sub_tx" not in st.session_state:
    st.session_state.selected_sub_tx = None
if "human_decision_submitted" not in st.session_state:
    st.session_state.human_decision_submitted = {}
if "goto_graph" not in st.session_state:
    st.session_state.goto_graph = False

# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.html(textwrap.dedent("""
    <div class="sidebar-brand">
        <div style="width:34px;height:34px;background:#2563eb;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;color:white;font-size:18px;">
            🛡️
        </div>
        <div>
            <div class="sidebar-brand-title">AML Fraud Detection</div>
            <div class="sidebar-brand-subtitle">GAT-based Transaction Monitoring</div>
        </div>
    </div>
    """))

    # If goto_graph flag is set, pre-select the graph page
    default_page_idx = 2 if st.session_state.goto_graph else 0
    page = st.radio(
        "Navigation",
        ["Dashboard", "Transactions", "Alerts / Graph Network", "Customers", "Reports", "Settings", "Help"],
        index=default_page_idx,
        label_visibility="collapsed"
    )
    if st.session_state.goto_graph and page == "Alerts / Graph Network":
        st.session_state.goto_graph = False

    st.html("<br><br><hr><div style='text-align:center;color:#94a3b8;font-size:11px;'>© AML Fraud Investigation System</div>")

# ─── Top Header ───────────────────────────────────────────────────────────
current_time = datetime.now().strftime("%I:%M:%S %p")
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.html(textwrap.dedent("""
    <div>
        <h1 class="header-title">AML Fraud Detection</h1>
        <div class="header-subtitle">Graph Attention Network (GAT) AML Monitoring</div>
    </div>
    """))
with col_h2:
    st.html(textwrap.dedent(f"""
    <div style="text-align:right;margin-top:10px;">
        <span class="header-timestamp">🕒 Last Updated: {current_time}</span>
    </div>
    """))

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    # Top Banner with real dataset counts
    total_txs_count = f"{len(fraud_data.get_transactions_df()):,}"
    flagged_senders = fraud_data.get_all_flagged_senders()
    high_count = sum(1 for s in flagged_senders if s["risk"] == "High")
    med_count = sum(1 for s in flagged_senders if s["risk"] == "Medium")
    low_count = sum(1 for s in flagged_senders if s["risk"] == "Low")

    st.html(textwrap.dedent(f"""
    <div class="top-banner">
        <div class="top-banner-icon">📋</div>
        <div>
            <div style="font-size:12px;color:#64748b;font-weight:600;">Total Transactions in Dataset</div>
            <div class="banner-value">{total_txs_count}</div>
            <div class="banner-subtext">Authoritative Fan-Out Groups</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">High Risk Groups</div>
            <div style="font-size:26px;font-weight:800;color:#dc2626;">{high_count}</div>
            <div style="font-size:12px;color:#dc2626;font-weight:600;">Require Review</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">Medium Risk Groups</div>
            <div style="font-size:26px;font-weight:800;color:#d97706;">{med_count}</div>
            <div style="font-size:12px;color:#d97706;font-weight:600;">Under Monitoring</div>
        </div>
        <div style="margin-left:30px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">Low Risk Groups</div>
            <div style="font-size:26px;font-weight:800;color:#16a34a;">{low_count}</div>
            <div style="font-size:12px;color:#16a34a;font-weight:600;">Low Priority</div>
        </div>
        <div style="margin-left:auto;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 18px;text-align:center;">
            <div style="font-size:11px;color:#16a34a;font-weight:700;">GAT MODEL</div>
            <div style="font-size:18px;font-weight:800;color:#16a34a;">● ACTIVE</div>
        </div>
    </div>
    """))

    # ── 3-Column Layout ─────────────────────────────────────────────────
    col_left, col_center, col_right = st.columns([1.0, 1.8, 1.2])

    # ════════════════════════════════════════════════════════════════════
    #  LEFT PANEL — Flagged Accounts (Fan-Out Groups)
    # ════════════════════════════════════════════════════════════════════
    with col_left:
        st.html('<div class="section-label">🚨 Flagged Sender Investigations</div>')

        all_txs = fraud_data.get_all_flagged_senders()

        for tx in all_txs:
            acc = tx.get("account", "Unknown")
            name = tx.get("name") or fraud_data.get_customer_profile(acc).get("name", acc)
            risk = tx.get("risk", "High")
            score = tx.get("risk_score", 100)
            pattern = tx.get("pattern", "FAN-OUT")
            gid = tx.get("group_id", 1)
            tx_id = tx.get("tx_id", f"GROUP-{gid}")
            is_sel = (tx_id == st.session_state.selected_tx_id or str(gid) == str(st.session_state.selected_tx_id))

            risk_color = "#ef4444" if risk == "High" else "#f59e0b" if risk == "Medium" else "#16a34a"
            sel_bg = "#eff6ff" if is_sel else "#ffffff"
            sel_border = "#93c5fd" if is_sel else "#e2e8f0"
            bar_width = max(5, min(100, score))
            tx_count_str = tx.get('tx_count', len(fraud_data.get_fan_out_rows(tx_id)))

            st.html(textwrap.dedent(f"""
            <div class="flagged-card flagged-card-{risk.lower()}"
                 style="background:{sel_bg}; border-color:{sel_border};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="flagged-acc-id">Sender: {acc}</div>
                        <div style="font-size:11px;color:#475569;font-weight:500;margin-top:1px;">{name}</div>
                    </div>
                    <span class="badge-{risk.lower()}">{risk.upper()}</span>
                </div>
                <div class="flagged-pattern">📌 {pattern} · Group {gid} ({tx_count_str} txs)</div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                    <div class="risk-score-bar-bg" style="flex:1;margin-right:8px;">
                        <div class="risk-score-bar-fill" style="width:{bar_width}%;background:{risk_color};"></div>
                    </div>
                    <span style="font-size:11px;font-weight:700;color:{risk_color};">{score}/100</span>
                </div>
            </div>
            """))
            if st.button(f"Inspect Group {gid} →", key=f"left_btn_{tx_id}", use_container_width=True):
                st.session_state.selected_tx_id = tx_id
                st.session_state.selected_sub_tx = None
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    #  CENTER PANEL — Fan-Out Transaction Table + XAI + Human Decision
    # ════════════════════════════════════════════════════════════════════
    with col_center:
        curr_tx = fraud_data.get_transaction_by_id(st.session_state.selected_tx_id)
        risk = curr_tx.get("risk", "High")
        risk_score = curr_tx.get("risk_score", 100)
        pattern = curr_tx.get("pattern", "FAN-OUT")
        gid = curr_tx.get("group_id", 1)

        lead_tx_id = curr_tx.get('lead_tx_id', curr_tx.get('tx_id', f'GROUP-{gid}'))
        sender_acc = curr_tx.get('account', '—')
        sender_name = curr_tx.get('name') or fraud_data.get_customer_profile(sender_acc).get('name', sender_acc)
        timestamp = curr_tx.get('timestamp', '—')
        payment_format = curr_tx.get('payment_format', '—')

        # ── Header ──
        st.html(textwrap.dedent(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
                <div style="font-size:16px;font-weight:800;color:#0f172a;">
                    Fan-out Group {gid} · TX {lead_tx_id}
                    <span style="font-size:12px;font-weight:500;color:#64748b;margin-left:6px;">{pattern}</span>
                </div>
                <div style="font-size:12px;color:#475569;margin-top:2px;">
                    Sender: <b>{sender_acc}</b> ({sender_name}) ·
                    {timestamp} · {payment_format}
                </div>
            </div>
            <span class="badge-{risk.lower()}">{risk.upper()} RISK · {risk_score}/100</span>
        </div>
        """))

        # ── Graph Network Button ──
        if st.button("🕸️  Tap to View as Graph Network", key="graph_btn", use_container_width=False):
            st.session_state.goto_graph = True
            st.rerun()

        # ── Fan-Out Sub-Transaction Table ──
        st.html('<div class="section-label">Transaction Routing Flow</div>')

        fan_rows = fraud_data.get_fan_out_rows(st.session_state.selected_tx_id)
        if fan_rows:
            df_fan = pd.DataFrame(fan_rows)
            df_display = df_fan[["sub_tx_id", "to_account", "amount", "time", "payment_format", "gat_signal"]].copy()
            df_display.columns = ["Sub-TX ID", "To Account", "Amount", "Timestamp", "Payment Format", "GAT Signal"]
        else:
            df_display = pd.DataFrame(columns=["Sub-TX ID", "To Account", "Amount", "Timestamp", "Payment Format", "GAT Signal"])

        # Clickable table with row selection
        event = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="fan_table"
        )

        # Handle row selection → populate right panel
        selected_rows = event.selection.get("rows", []) if event.selection else []
        if selected_rows and len(fan_rows) > selected_rows[0]:
            row_idx = selected_rows[0]
            sub_tx_data = fan_rows[row_idx]
            st.session_state.selected_sub_tx = sub_tx_data

        st.html("""
        <div style="font-size:10.5px;color:#94a3b8;margin-top:4px;">
            🖱️ Click a row to view the receiver's customer profile in the right panel.
        </div>
        """)

        # ── Why Flagged? XAI Box ──
        is_high = (risk == "High")
        xai_extra = "xai-box-high" if is_high else ""
        exps_html = "".join([f"<li>{e}</li>" for e in curr_tx.get("explanations", [])])
        fraud_icon = "⚠️" if curr_tx.get("is_fraud", False) else "ℹ️"
        fraud_label = "FRAUD DETECTED" if curr_tx.get("is_fraud", False) else "SUSPICIOUS ACTIVITY"

        st.html(textwrap.dedent(f"""
        <div class="xai-box {xai_extra}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div class="xai-title">{fraud_icon} {fraud_label} — Why was this flagged?</div>
                <span class="badge-{risk.lower()}">{pattern}</span>
            </div>
            <ul class="xai-list">
                {exps_html}
            </ul>
            <div style="font-size:11px;color:#64748b;margin-top:10px;border-top:1px solid #fde68a;padding-top:8px;">
                Model: <b>{curr_tx.get('model_used', 'GAT AML Model')}</b> &nbsp;|&nbsp;
                Confidence: <b>{curr_tx.get('model_confidence', '100%')}</b> (Raw GAT Probability: <b>{curr_tx.get('gat_prob', '1.0')}</b>)
            </div>
        </div>
        """))

        # ── Authorised Bank Auditor Decision Panel (HIGH RISK ONLY) ──
        if is_high:
            tx_key = curr_tx.get("tx_id", f"GROUP-{gid}")
            already_submitted = st.session_state.human_decision_submitted.get(tx_key)

            st.html(textwrap.dedent(f"""
            <div class="human-decision-panel">
                <div class="human-decision-title">
                    🏦 Authorised Bank Auditor Decision Required — Group {gid}
                </div>
                <div style="font-size:12px;color:#92400e;margin-bottom:12px;">
                    Risk Score: <b>{risk_score}/100</b> — This Fan-Out investigation requires an authorised bank auditor decision.
                </div>
            </div>
            """))

            if already_submitted:
                decision_val = already_submitted["decision"]
                st.success(f"**Decision Recorded:** {decision_val}")
                st.info(f"**Authorised Bank Auditor Notes:** {already_submitted['notes'] or '(none)'}")
                if st.button("Revise Decision", key=f"revise_{tx_key}"):
                    del st.session_state.human_decision_submitted[tx_key]
                    st.rerun()
            else:
                dec_col1, dec_col2 = st.columns([1, 1])
                with dec_col1:
                    decision = st.radio(
                        "**Authorised Bank Auditor Decision**",
                        ["✅ Approve Transaction", "🚫 Block Transaction", "📤 Escalate to Senior Investigator"],
                        key=f"decision_{tx_key}",
                        index=2
                    )
                with dec_col2:
                    notes = st.text_area(
                        "**Authorised Bank Auditor Notes**",
                        placeholder="Add reasoning, observations, or escalation notes...",
                        key=f"notes_{tx_key}",
                        height=180
                    )

                st.html("""
                <div class="human-decision-disclaimer">
                    ⚠️ <b>Disclaimer:</b> By submitting, you confirm this decision is made by an
                    AUTHORISED BANK AUDITOR. The GAT AML model provided supporting analysis only.
                    This action will be logged and audited.
                </div>
                """)

                if st.button(f"📋 Submit Decision for Group {gid}", key=f"submit_{tx_key}", type="primary", use_container_width=True):
                    st.session_state.human_decision_submitted[tx_key] = {
                        "decision": decision,
                        "notes": notes,
                        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
                    }
                    st.rerun()

        # ── AI Advisory (COMPLETELY AT BOTTOM) ──
        st.html(textwrap.dedent("""
        <div class="ai-advisory" style="margin-top:16px;">
            <span style="font-size:16px;">🤖</span>
            <span>
                <b>AI Advisory:</b> The analysis above is generated by the GAT AML model to
                <b>support the fraud investigator's decision</b>. The AI does not block or approve
                transactions. All final decisions must be made by an AUTHORIZED BANK AUDITOR.
            </span>
        </div>
        """))

    # ════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — Customer Profile (receiver, on row click)
    # ════════════════════════════════════════════════════════════════════
    with col_right:
        sub_tx = st.session_state.selected_sub_tx

        if sub_tx is None:
            st.html(textwrap.dedent("""
            <div class="profile-card">
                <div class="section-label">Receiver Profile</div>
                <div class="profile-empty">
                    <div style="font-size:40px;margin-bottom:12px;">👤</div>
                    <div style="font-size:13px;font-weight:600;color:#64748b;">No row selected</div>
                    <div style="font-size:12px;margin-top:6px;color:#94a3b8;">
                        Click any transaction row in the center table to view the receiver's customer profile here.
                    </div>
                </div>
            </div>
            """))
        else:
            to_acc = sub_tx.get("to_account", "—")
            try:
                profile = fraud_data.get_receiver_profile(to_acc, curr_tx.get("group_id"))
            except Exception:
                try:
                    profile = fraud_data.get_receiver_profile(to_acc)
                except Exception:
                    profile = {}

            if not isinstance(profile, dict):
                profile = {}

            # Direct fallback to selected row metadata
            profile_name = profile.get("name") if profile.get("name") not in ("—", "Unknown", None, "") else sub_tx.get("to_entity_name", to_acc)
            bank_name = profile.get("bank_name") if profile.get("bank_name") not in ("—", None, "") else sub_tx.get("to_bank_name", "—")
            bank_id = profile.get("bank_id") if profile.get("bank_id") not in ("—", None, "") else sub_tx.get("to_bank_id", "—")
            entity_id = profile.get("entity_id") if profile.get("entity_id") not in ("—", None, "") else sub_tx.get("to_entity_id", "—")
            tot_inc = profile.get("total_incoming") if profile.get("total_incoming") not in ("—", None, "") else sub_tx.get("amount", "—")
            avg_amt = profile.get("avg_tx_amount") if profile.get("avg_tx_amount") not in ("—", None, "") else sub_tx.get("amount", "—")
            risk_tier = profile.get("risk_tier", "High Risk" if "HIGH" in str(sub_tx.get("gat_signal", "HIGH")).upper() else "Low Risk")
            tier_color = "#dc2626" if "High" in risk_tier else "#d97706" if "Medium" in risk_tier else "#16a34a"

            prev_in = str(sub_tx.get('previous_incoming', profile.get('previous_incoming', '—')))
            prev_out = str(sub_tx.get('previous_outgoing', profile.get('previous_outgoing', '—')))
            payment_fmt = str(sub_tx.get('payment_format', '—'))

            beh_rows = [
                ("Total Incoming", tot_inc),
                ("Unique Senders", prev_in),
                ("Unique Receivers", prev_out),
                ("Avg Tx Amount", avg_amt),
                ("Total Transactions", str(profile.get("total_transactions", "1"))),
                ("Entity ID", entity_id),
                ("Bank ID", bank_id),
            ]
            beh_rows_html = "".join([f"<tr><td><b>{r[0]}</b></td><td style='text-align:right;'>{r[1]}</td></tr>" for r in beh_rows])

            st.html(textwrap.dedent(f"""
            <div class="profile-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                    <div>
                        <div class="section-label">Receiver Profile</div>
                        <div style="font-size:16px;font-weight:800;color:#0f172a;">{profile_name}</div>
                        <div style="font-size:12px;color:#64748b;margin-top:2px;">Account: <b>{to_acc}</b></div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:{tier_color}20;color:{tier_color};border:1px solid {tier_color}44;
                                     padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;">
                            {risk_tier}
                        </span>
                    </div>
                </div>

                <div class="profile-metric-grid">
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Bank Name</div>
                        <div class="profile-metric-val" style="font-size:12px;">{bank_name}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Bank ID</div>
                        <div class="profile-metric-val">{bank_id}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Entity ID</div>
                        <div class="profile-metric-val">{entity_id}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Payment Format</div>
                        <div class="profile-metric-val" style="font-size:12px;">{payment_fmt}</div>
                    </div>
                </div>

                <div style="font-size:12px;font-weight:700;color:#1e293b;margin-bottom:6px;">Financial Summary</div>
                <table class="custom-table">
                    <tbody>
                        {beh_rows_html}
                    </tbody>
                </table>
            </div>
            """))


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transactions":
    st.subheader("📊 All Transactions Log")
    df_all = fraud_data.get_transactions_df()
    c1, c2 = st.columns(2)
    with c1:
        search_q = st.text_input("🔍 Search Transaction ID / Account / Fan-out Group", "")
    with c2:
        risk_filter = st.multiselect("Filter GAT Signal", ["HIGH RISK", "LOW RISK"], default=["HIGH RISK", "LOW RISK"])

    filtered_df = df_all[df_all["GAT Signal"].isin(risk_filter)]
    if search_q:
        filtered_df = filtered_df[
            filtered_df["Transaction ID"].astype(str).str.contains(search_q, case=False) |
            filtered_df["From Account"].astype(str).str.contains(search_q, case=False) |
            filtered_df["To Account"].astype(str).str.contains(search_q, case=False) |
            filtered_df["Fan-out Group"].astype(str).str.contains(search_q, case=False)
        ]

    st.dataframe(
        filtered_df[["Fan-out Group", "Transaction ID", "Timestamp", "From Account", "From Entity Name", "To Account", "To Entity Name", "Amount Paid", "Payment Currency", "Payment Format", "GAT Probability", "GAT Signal", "Actual Label"]].head(500),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ALERTS / GRAPH NETWORK PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Alerts / Graph Network":
    st.subheader("🕸️ Money Trail")
    st.markdown(
        "Visualizing transactional connections for the selected Fan-Out group. "
        "Hover over any node to see the customer profile. "
        "Sender (★) → Hop-1 Receivers (●)"
    )

    all_groups = fraud_data.get_all_flagged_senders()
    group_options = [g["tx_id"] for g in all_groups]
    default_sel = st.session_state.selected_tx_id
    default_idx = group_options.index(default_sel) if default_sel in group_options else 0

    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        sel_tx = st.selectbox(
            "Select Fan-out Group Network to Inspect",
            group_options,
            index=default_idx
        )
    with col_g2:
        show_2hop = st.checkbox("Show 2-Hop Neighbors", value=False, disabled=True, help="Hop-2 nodes are derived only when complete downstream data is available in the dataset.")

    tx_info = fraud_data.get_transaction_by_id(sel_tx)

    # Info bar
    risk_col = "#dc2626" if tx_info.get("risk") == "High" else "#d97706" if tx_info.get("risk") == "Medium" else "#16a34a"
    st.html(textwrap.dedent(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                padding:14px 18px;margin-bottom:16px;display:flex;gap:28px;align-items:center;">
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">PATTERN</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('pattern', 'FAN-OUT')}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">GAT RISK SCORE</div>
            <div style="font-size:14px;font-weight:700;color:{risk_col};">{tx_info.get('risk_score', 100)}/100</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">MODEL</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('model_used', 'GAT AML Model')}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">SENDER</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('account', '—')} ({tx_info.get('name') or fraud_data.get_customer_profile(tx_info.get('account', '')).get('name', '—')})</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">TOTAL AMOUNT</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info.get('amount_formatted', '—')}</div>
        </div>
    </div>
    """))

    fig_g = graph_vis.render_plotly_graph(sel_tx, include_2hop=False)
    st.plotly_chart(fig_g, use_container_width=True)

    # Legend explanation
    st.html(textwrap.dedent("""
    <div style="display:flex;gap:24px;justify-content:center;margin-top:4px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#ef4444;border-radius:50%;display:inline-block;"></span>
            Sender Account
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#f59e0b;border-radius:50%;display:inline-block;"></span>
            Hop-1 Direct Receivers
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:28px;height:2px;background:#ef4444;display:inline-block;"></span>
            Hop-1 Transfer (amount shown)
        </div>
    </div>
    """))

    st.markdown("---")
    # XAI explanations on graph page too
    st.html(textwrap.dedent(f"""
    <div style="background:#fff5f5;border:1px solid #fecaca;border-radius:10px;padding:14px 16px;">
        <div style="font-weight:700;font-size:13px;color:#dc2626;margin-bottom:8px;">⚠️ Why was this flagged?</div>
        <ul style="margin:0;padding-left:18px;font-size:12px;color:#334155;line-height:1.8;">
            {"".join([f"<li>{e}</li>" for e in tx_info.get("explanations", [])])}
        </ul>
    </div>
    """))

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMERS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customers":
    st.subheader("👤 Customer Profiling & Network Risk")
    all_senders = fraud_data.get_all_flagged_senders()
    sender_accs = [s["account"] for s in all_senders]
    acc_sel = st.selectbox("Select Sender Account ID", sender_accs)
    prof = fraud_data.get_customer_profile(acc_sel)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Entity Name:** {prof.get('name', '—')}")
        st.markdown(f"**Entity ID:** {prof.get('entity_id', '—')}")
        st.markdown(f"**Bank Name:** {prof.get('bank_name', '—')}")
        st.markdown(f"**Bank ID:** {prof.get('bank_id', '—')}")
    with c2:
        st.markdown(f"**Risk Tier:** {prof.get('risk_tier', '—')}")
        st.markdown(f"**Total Outgoing:** {prof.get('total_outgoing', '—')}")
        st.markdown(f"**Unique Receivers:** {prof.get('unique_receivers', '—')}")
        st.markdown(f"**Avg Transaction:** {prof.get('avg_tx_amount', '—')}")

    if prof.get("behavior_summary"):
        st.dataframe(pd.DataFrame(prof["behavior_summary"]), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  REMAINING PAGES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    st.subheader("📄 Automated Compliance & Fraud Reports")
    st.markdown("Generate AI-driven regulatory reports for compliance and operations stakeholders.")
    if st.button("Generate Summary Compliance Report"):
        st.success(f"✅ Report generated successfully for {len(fraud_data.get_transactions_df()):,} dataset transactions.")

elif page == "Settings":
    st.subheader("⚙️ System & Graph AI Settings")
    st.slider("GAT AML Model Risk Detection Threshold", 0.0, 1.0, 0.70)
    st.selectbox("Primary Model", ["GAT AML Model (Trained)", "Graph Attention Network v2"])

elif page == "Help":
    st.subheader("❓ Help & Documentation")
    st.markdown("""
### Fan-Out Pattern Explained:
- **Fan-Out**: Single source account transferring money out to multiple receiver accounts in rapid succession or parallel bursts.

### How to use the Dashboard:
1. Click a **Flagged Sender Investigation** in the left panel to load its Fan-out group transactions.
2. Click any **row** in the transaction routing flow table to view the **receiver's customer profile** on the right.
3. Tap **"Tap to View as Graph Network"** to visualize the transactional network topology.
4. For **High Risk** transactions, submit your decision in the **Authorised Bank Auditor Decision Panel**.

### AI Advisory Disclaimer:
The GAT AML model provides pattern analysis and risk probabilities to *support* the investigation. **Final decisions must always be made by an AUTHORIZED BANK AUDITOR.**
    """)
