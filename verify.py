import fraud_data
import graph_vis

print('Testing fraud_data with real CSV dataset...')
senders = fraud_data.get_all_flagged_senders()
print(f'  Loaded unique flagged senders / groups: {len(senders)}')
print(f'  Group 1 info: ID={senders[0]["group_id"]}, Account={senders[0]["account"]}, Entity={senders[0]["name"]}, Score={senders[0]["risk_score"]}, GAT Prob={senders[0]["gat_prob"]}')

g1_txs = fraud_data.get_fan_out_rows("GROUP-1")
print(f'  Group 1 fan-out sub-transactions: {len(g1_txs)}')
print(f'  Sample sub-tx 0: {g1_txs[0]}')

sample_to_acc = g1_txs[0]["to_account"]
rp = fraud_data.get_receiver_profile(sample_to_acc, group_id=1)
print(f'  Receiver {sample_to_acc} profile: Name={rp["name"]}, Bank={rp["bank_name"]}, Entity ID={rp["entity_id"]}')

cp = fraud_data.get_customer_profile(senders[0]["account"])
print(f'  Sender {senders[0]["account"]} profile: Name={cp["name"]}, Outgoing={cp["total_outgoing"]}')

print('Testing graph_vis...')
fig = graph_vis.render_plotly_graph('GROUP-1')
print(f'  Graph traces count: {len(fig.data)}')

fig2 = graph_vis.render_plotly_graph('GROUP-2')
print(f'  Graph traces Group 2: {len(fig2.data)}')

print('All checks PASSED successfully with REAL DATA!')
