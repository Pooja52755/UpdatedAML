"""
Graph Visualization Module using Plotly & NetworkX
Renders interactive graph network topology for AML transaction analysis.
Visualizes real Sender (★) → Hop-1 Receivers (●) with real transaction amounts.
"""

import plotly.graph_objects as go
import networkx as nx
import fraud_data


def _build_hover_text(node, data):
    """Build a rich hover tooltip for a given node using real dataset fields."""
    node_type = data.get("node_type", "node")
    hop = data.get("hop", 0)

    if hop == 0:
        # Source sender node
        entity_name = data.get("entity_name") or node
        bank_name = data.get("bank_name", "—")
        bank_id = data.get("bank_id", "—")
        entity_id = data.get("entity_id", "—")
        gat_signal = data.get("gat_signal", "—")
        gat_prob = data.get("gat_prob", "—")

        return (
            f"<b>🔴 SENDER ACCOUNT</b><br>"
            f"<b>{entity_name}</b><br>"
            f"Account: {node}<br>"
            f"Entity ID: {entity_id}<br>"
            f"Bank: {bank_name} (ID: {bank_id})<br>"
            f"GAT Signal: {gat_signal}<br>"
            f"GAT Probability: {gat_prob}"
        )
    elif hop == 1:
        # First-hop receiver
        entity_name = data.get("entity_name") or node
        bank_name = data.get("bank_name", "—")
        bank_id = data.get("bank_id", "—")
        entity_id = data.get("entity_id", "—")
        amount = data.get("amount", "—")

        return (
            f"<b>🟡 HOP-1 RECEIVER</b><br>"
            f"<b>{entity_name}</b><br>"
            f"Account: {node}<br>"
            f"Amount Received: {amount}<br>"
            f"Entity ID: {entity_id}<br>"
            f"Bank: {bank_name} (ID: {bank_id})"
        )
    elif hop == 2:
        return (
            f"<b>⬡ HOP-2 NODE</b><br>"
            f"Account: {node}"
        )
    else:
        return f"Account: {node}<br>Role: {node_type.capitalize()}"


def render_plotly_graph(tx_id_or_group_id, include_2hop=False):
    """
    Renders an interactive Plotly figure representing the transaction graph.
    Shows source → hop-1 receivers with actual transfer amounts.
    Hovering on any node shows the real customer & bank profile.
    """
    G = fraud_data.create_network_graph(tx_id_or_group_id, include_2hop=include_2hop)
    
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No graph data available",
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            height=520,
        )
        return fig

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # ── Edge traces by hop ─────────────────────────────────────────────────
    edge_traces = []
    annotation_list = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        hop = edge[2].get("hop", 1)
        amount = edge[2].get("amount", "Transfer")

        color = "#ef4444" if hop == 1 else "#e2e8f0"
        width = 2.5 if hop == 1 else 0.8
        dash = "solid" if hop == 1 else "dot"

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color, dash=dash),
            hoverinfo="text",
            hovertext=f"<b>Transfer:</b> {edge[0]} ➔ {edge[1]}<br><b>Amount:</b> {amount}",
            showlegend=False,
        ))

        # Mid-point label for hop-1 edges
        if hop == 1:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            annotation_list.append(dict(
                x=mx, y=my,
                text=f"<b>{amount}</b>",
                showarrow=False,
                font=dict(size=9, color="#dc2626"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#fecaca",
                borderwidth=1,
                borderpad=2,
            ))

    # ── Node traces by type ────────────────────────────────────────────────
    node_groups = {"source": [], "hop1": [], "hop2": []}

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        hop = data.get("hop", 1)
        key = "source" if hop == 0 else ("hop2" if hop == 2 else "hop1")
        node_groups[key].append((node, data, x, y))

    node_traces = []

    group_styles = {
        "source": dict(
            name="Sender Account",
            symbol="star",
            size=38,
            color_fn=lambda d: d.get("color", "#ef4444"),
            line_color="#ffffff",
            line_width=3,
            text_color="#1e293b",
            opacity=0.95,
        ),
        "hop1": dict(
            name="Hop-1 Receivers",
            symbol="circle",
            size=28,
            color_fn=lambda d: d.get("color", "#f59e0b"),
            line_color="#ffffff",
            line_width=2,
            text_color="#334155",
            opacity=0.95,
        ),
        "hop2": dict(
            name="Hop-2 Nodes",
            symbol="diamond",
            size=15,
            color_fn=lambda d: "#f1f5f9",
            line_color="#cbd5e1",
            line_width=1,
            text_color="#94a3b8",
            opacity=0.65,
        ),
    }

    for key, nodes in node_groups.items():
        if not nodes:
            continue
        style = group_styles[key]
        xs = [n[2] for n in nodes]
        ys = [n[3] for n in nodes]
        labels = [n[0] for n in nodes]
        hover = [_build_hover_text(n[0], n[1]) for n in nodes]
        colors = [style["color_fn"](n[1]) for n in nodes]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=style["name"],
            text=labels,
            textposition="top center",
            textfont=dict(size=9, color=style["text_color"]),
            hoverinfo="text",
            hovertext=hover,
            hoverlabel=dict(
                bgcolor="#1e293b",
                font_size=11,
                font_color="white",
                bordercolor="#475569",
            ),
            marker=dict(
                symbol=style["symbol"],
                size=style["size"],
                color=colors,
                line=dict(width=style["line_width"], color=style["line_color"]),
                opacity=style["opacity"],
            ),
            showlegend=True,
        ))

    all_traces = edge_traces + node_traces
    title_text = f"<b>Transaction Network — {tx_id_or_group_id}</b>"

    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=11),
            ),
            hovermode="closest",
            margin=dict(b=30, l=20, r=20, t=50),
            annotations=annotation_list,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            height=520,
            title=dict(
                text=f"{title_text}   "
                     f"<span style='font-size:12px;color:#64748b;'>"
                     f"★ Sender  🟡 Hop-1 Receivers  "
                     f"— Hover any node for customer profile</span>",
                font=dict(size=14, color="#1e293b"),
                x=0.0,
            ),
        )
    )

    return fig
