import pickle

with open('./log_st/histories_st.pkl', 'rb') as f:
	data = pickle.load(f)

lt = data['loss_history']
min_loss = min(lt.items(), key=lambda lt: lt[1])
print(min_loss)

