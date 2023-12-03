import pickle

policies = []
ps = ["act_", "obs_", "", "static_", "dynamic_"]
for p in ps:
    with open(p+"policy", 'rb') as f:
        policies.append(pickle.load(f))
