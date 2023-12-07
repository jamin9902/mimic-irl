import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os, pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
from itertools import product
import sys, os
import trajectory as T                      # trajectory generation
import optimizer as O                       # stochastic gradient descent optimizer
import solver as S                          # MDP solver (value-iteration)
import plot as P
import suboptimality as SO
import pickle


num_data = 355504
np.random.seed(66)

"""
def to_interval(istr):
    c_left = istr[0]=='['
    c_right = istr[-1]==']'
    closed = {(True, False): 'left',
              (False, True): 'right',
              (True, True): 'both',
              (False, False): 'neither'
              }[c_left, c_right]
    left, right = map(pd.to_datetime, istr[1:-1].split(','))
    return pd.Interval(left, right, closed)

re_split = False
frac = [0.4,0.2,0.4]
assert np.sum(frac) == 1
frac = np.cumsum(frac)
#print (frac)
data_save_path= 'data/'

def sliding(gs, window_size = 6):
    npr_l = []
    for g in gs:
        npr = np.concatenate([np.zeros([window_size-1, g.shape[1]]),g])
        npr_l.append(sliding_window_view(npr, (window_size, g.shape[1])).squeeze(1))
    return np.vstack(npr_l)

aggr_df = pd.read_csv('mimic_iv_hypotensive_cut2.csv',sep = ',', header = 0,converters={1:to_interval}).set_index(['stay_id','time']).sort_index()
# create action bins (four actions in total)
aggr_df['action'] = aggr_df['bolus(binary)']*2 + aggr_df['vaso(binary)']
all_idx = np.random.permutation(aggr_df.index.get_level_values(0).unique())
train_df = aggr_df.loc[all_idx[:int(len(all_idx)*frac[0])]].sort_index()
test_df = aggr_df.loc[all_idx[int(len(all_idx)*frac[0]):int(len(all_idx)*frac[1])]].sort_index()
valid_df = aggr_df.loc[all_idx[int(len(all_idx)*frac[1]):]].sort_index()
# print (np.unique(train_df['action'],return_counts=True)[1]*1./len(train_df))
# pickle.dump([train_df, test_df, valid_df], open(data_save_path+'processed_mimic_hyp_2.pkl','wb'))
drop_columns = ['vaso(amount)','bolus(amount)',\
            'any_treatment(binary)','vaso(binary)','bolus(binary)']

# for now drop indicators about bolus and vaso
train_df = train_df.drop(columns=drop_columns)
test_df = test_df.drop(columns=drop_columns)
valid_df = valid_df.drop(columns=drop_columns)

#### imputation
impute_table = pd.read_csv('mimic_iv_hypotensive_cut2_impute_table.csv',sep=',',header=0).set_index(['feature'])
train_df = train_df.fillna(method='ffill')
test_df = test_df.fillna(method='ffill')
valid_df = valid_df.fillna(method='ffill')

for f in impute_table.index:
    train_df[f] = train_df[f].fillna(value = impute_table.loc[f].values[0])
    test_df[f] = test_df[f].fillna(value = impute_table.loc[f].values[0])
    valid_df[f] = valid_df[f].fillna(value = impute_table.loc[f].values[0])

with open("train_df","wb") as f:
    pickle.dump(train_df,f)
with open("test_df","wb") as f:
    pickle.dump(test_df,f)
with open("valid_df","wb") as f:
    pickle.dump(valid_df,f)    
"""

"""
with open("train_df","rb") as f:
    train_df = pickle.load(f)
with open("test_df","rb") as f:
    test_df = pickle.load(f)
with open("valid_df","rb") as f:
    valid_df = pickle.load(f)

#OBSERVATION AMBIGUITY
suboptimal_features = ['creatinine', 'fraction_inspired_oxygen', 'lactate', 'urine_output',
                   'alanine_aminotransferase', 'asparate_aminotransferase',
                   'mean_blood_pressure', 'diastolic_blood_pressure',
                   'systolic_blood_pressure', 'gcs', 'partial_pressure_of_oxygen', 
                   'heart_rate', 'temperature', 'respiratory_rate']

obs_train_df = SO.observation_ambiguity(train_df, suboptimal_features, 0.1, 0.3)
obs_test_df = SO.observation_ambiguity(test_df, suboptimal_features, 0.1, 0.3)
obs_valid_df = SO.observation_ambiguity(valid_df, suboptimal_features, 0.1, 0.3)

with open("obs_train_df","wb") as f:
    pickle.dump(obs_train_df,f)
with open("obs_test_df","wb") as f:
    pickle.dump(obs_test_df,f)
with open("obs_valid_df","wb") as f:
    pickle.dump(obs_valid_df,f)    

# ACTION AMBIGUITY
act_train_df = SO.action_ambiguity(train_df, 0.3)
act_test_df = SO.action_ambiguity(test_df, 0.3)
act_valid_df = SO.action_ambiguity(valid_df, 0.3)

with open("act_train_df","wb") as f:
    pickle.dump(act_train_df,f)
with open("act_test_df","wb") as f:
    pickle.dump(act_test_df,f)
with open("act_valid_df","wb") as f:
    pickle.dump(act_valid_df,f)    
"""

"""
#ambigs = ["obs_", "act_", ""]
ambigs=["obs_"]
for ambig in ambigs:
    with open("data/"+ambig+"train_df", 'rb') as f:
        train_df = pickle.load(f)
    with open("data/"+ambig+"test_df", 'rb') as f:
        test_df = pickle.load(f)
    with open("data/"+ambig+"valid_df", 'rb') as f:
        valid_df = pickle.load(f)

    data_non_normalized_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=False).head(num_data).copy()

    #### standard normalization ####
    normalize_features = ['creatinine', 'fraction_inspired_oxygen', 'lactate', 'urine_output',
                      'alanine_aminotransferase', 'asparate_aminotransferase',
                      'mean_blood_pressure', 'diastolic_blood_pressure',
                      'systolic_blood_pressure', 'gcs', 'partial_pressure_of_oxygen']
    mu, std = (train_df[normalize_features]).mean().values,(train_df[normalize_features]).std().values
    train_df[normalize_features] = (train_df[normalize_features] - mu)/std
    test_df[normalize_features] = (test_df[normalize_features] - mu)/std
    valid_df[normalize_features] = (valid_df[normalize_features] - mu)/std

    ### create data matrix ####
    X_train = train_df.loc[:,train_df.columns!='action']
    y_train = train_df['action']

    X_test = test_df.loc[:,test_df.columns!='action']
    y_test = test_df['action']

    X_valid = valid_df.loc[:, valid_df.columns!='action']
    y_valid = valid_df['action']

    X_df = pd.concat([X_train, X_valid, X_test], axis=0, ignore_index=True).copy()
    y_df = pd.concat([y_train, y_valid, y_test], axis=0, ignore_index=True).copy()

    data_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=False).copy()
    num_clusters = 100
    #kmeans = KMeans(n_clusters= num_clusters , random_state=0)
    #kmeans.fit(X_df)

    with open("data/"+ambig+"clusters",'rb') as f:
        kmeans = pickle.load(f)

    X_df['cluster'] = kmeans.labels_.copy()
    data_df['cluster'] = kmeans.labels_.copy()
    data_non_normalized_df['cluster'] = kmeans.labels_.copy()

    #static occlusion
    #excluded_states = [2, 89, 98, 54, 71, 49, 3, 97, 10, 74, 52, 58, 99, 6, 69,
    #                9, 94, 83, 85, 47, 20, 27, 64, 19, 22, 81, 92, 5, 35, 16]
    #data_df = SO.static_occlusion(data_df, excluded_states)

    #dynamic occlusion
    data_df = SO.dynamic_occlusion(data_df, 0.3)

    # Convert states and actions to one-hot encoding
    state_encoder = OneHotEncoder(sparse=False, categories= [np.arange(num_clusters)])
    action_encoder = OneHotEncoder(sparse=False, categories= [np.arange(4)])    

    states_onehot = state_encoder.fit_transform(X_df['cluster'].to_numpy().reshape(-1, 1))
    actions_onehot = action_encoder.fit_transform(y_df.to_numpy().reshape(-1, 1))

    # # Define neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(states_onehot.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(actions_onehot.shape[1], activation='softmax')  # Output layer with softmax for discrete actions
    ])

    # # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

    # # Train the model
    model.fit(states_onehot, actions_onehot,  epochs=5, batch_size=128)

    # # Evaluate the model
    test_loss = model.evaluate(states_onehot, actions_onehot)
    print("Test Loss:", test_loss)

    with open("data/clusters",'rb') as f:
        kmeans = pickle.load(f)
    with open("data/obs_clusters",'rb') as f:
        obs_kmeans = pickle.load(f)

    #bc_policy = np.argmax(model.predict(state_encoder.transform(np.arange(num_clusters).reshape(-1, 1))), axis =1)
    for i in range(100):
        bc_policy.append(np.argmax(model.predict(state_encoder.transform(obs_kmeans.predict(kmeans.cluster_centers_[i].reshape(1,-1)).reshape(1,-1)))))
    with open("dynamic_bc_policy", 'wb') as f:
        pickle.dump(bc_policy,f)
"""

"""
#ambigs = ["obs_","act_", ""]
ambigs = [""]
for ambig in ambigs:
    with open("data/"+ambig+"train_df", 'rb') as f:
        train_df = pickle.load(f)
    with open("data/"+ambig+"test_df", 'rb') as f:
        test_df = pickle.load(f)
    with open("data/"+ambig+"valid_df", 'rb') as f:
        valid_df = pickle.load(f)

    data_non_normalized_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=False).head(num_data).copy()

    #### standard normalization ####
    normalize_features = ['creatinine', 'fraction_inspired_oxygen', 'lactate', 'urine_output',
                      'alanine_aminotransferase', 'asparate_aminotransferase',
                      'mean_blood_pressure', 'diastolic_blood_pressure',
                      'systolic_blood_pressure', 'gcs', 'partial_pressure_of_oxygen']
    mu, std = (train_df[normalize_features]).mean().values,(train_df[normalize_features]).std().values
    train_df[normalize_features] = (train_df[normalize_features] - mu)/std
    test_df[normalize_features] = (test_df[normalize_features] - mu)/std
    valid_df[normalize_features] = (valid_df[normalize_features] - mu)/std

    ### create data matrix ####
    X_train = train_df.loc[:,train_df.columns!='action']
    y_train = train_df['action']

    X_test = test_df.loc[:,test_df.columns!='action']
    y_test = test_df['action']

    X_valid = valid_df.loc[:, valid_df.columns!='action']
    y_valid = valid_df['action']

    X_df = pd.concat([X_train, X_valid, X_test], axis=0, ignore_index=True).copy()
    y_df = pd.concat([y_train, y_valid, y_test], axis=0, ignore_index=True).copy()

    data_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=False).copy()
    num_clusters = 100
            
    with open("data/"+ambig+"clusters",'rb') as f:
        kmeans = pickle.load(f)
    with open("dynamic_counts",'wb') as f:
        pickle.dump(np.unique(kmeans.labels_, return_counts = True)[1], f)

    X_df['cluster'] = kmeans.labels_.copy()
    data_df['cluster'] = kmeans.labels_.copy()
    data_non_normalized_df['cluster'] = kmeans.labels_.copy()

    #static occlusion
    excluded_states = [2, 89, 98, 54, 71, 49, 3, 97, 10, 74, 52, 58, 99, 6, 69,
                    9, 94, 83, 85, 47, 20, 27, 64, 19, 22, 81, 92, 5, 35, 16]
    data_df = SO.static_occlusion(data_df, excluded_states)

    #dynamic occlusion
    data_df = SO.dynamic_occlusion(data_df, 0.3)
    with open("dynamic_data_df","wb") as f:
        pickle.dump(data_df,f)
    
    unique_stay_ids = data_df.index.get_level_values('stay_id').unique()
    trajectories = []

    for stay_id in unique_stay_ids:
        states, actions = data_df.loc[stay_id]['cluster'], data_df.loc[stay_id]['action']
        trajectory = []
        for i in range(len(states) - 1):
            trajectory.append((states[i], int(actions[i]), states[i+1] ))
        trajectories.append(T.Trajectory(trajectory))

    with open("dynamic_trajectories","wb") as f:
        pickle.dump(trajectories,f)

    terminal_states = []

    for traj in trajectories:
        terminal_states.append(traj._t[-1][-1])

    terminal_states = list(set(terminal_states))

    smoothing_value = 1
    p_transition = np.zeros((num_clusters, num_clusters, 4)) + smoothing_value
    for traj in trajectories:
        for tran in traj._t:
            p_transition[tran[0], tran[2], tran[1]] +=1

    p_transition = p_transition/ p_transition.sum(axis = 1)[:, np.newaxis, :]
    with open("dynamic_p_transition","wb") as f:
        pickle.dump(p_transition, f)
"""


#ambigs = ["obs_", "act_", ""]
#ambigs = [""]
ambigs = ["act_"]
num_clusters = 100
from maxent import irl, irl_causal
for ambig in ambigs:


    with open("data/"+ambig+"train_df", 'rb') as f:
    train_df = pickle.load(f)
with open("data/"+ambig+"test_df", 'rb') as f:
    test_df = pickle.load(f)
with open("data/"+ambig+"valid_df", 'rb') as f:
    valid_df = pickle.load(f)

data_non_normalized_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=False).head(num_data).copy()

#### standard normalization ####
normalize_features = ['creatinine', 'fraction_inspired_oxygen', 'lactate', 'urine_output',
                  'alanine_aminotransferase', 'asparate_aminotransferase',
                  'mean_blood_pressure', 'diastolic_blood_pressure',
                  'systolic_blood_pressure', 'gcs', 'partial_pressure_of_oxygen']
mu, std = (train_df[normalize_features]).mean().values,(train_df[normalize_features]).std().values
train_df[normalize_features] = (train_df[normalize_features] - mu)/std
test_df[normalize_features] = (test_df[normalize_features] - mu)/std
valid_df[normalize_features] = (valid_df[normalize_features] - mu)/std

### create data matrix ####
X_train = train_df.loc[:,train_df.columns!='action']
y_train = train_df['action']

X_test = test_df.loc[:,test_df.columns!='action']
y_test = test_df['action']

X_valid = valid_df.loc[:, valid_df.columns!='action']
y_valid = valid_df['action']

X_df = pd.concat([X_train, X_valid, X_test], axis=0, ignore_index=True).copy()
y_df = pd.concat([y_train, y_valid, y_test], axis=0, ignore_index=True).copy()

data_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=False).copy()
num_clusters = 100
kmeans = KMeans(n_clusters= num_clusters , random_state=0)
kmeans.fit(X_df)

#with open("data/"+ambig+"clusters",'rb') as f:
#    kmeans = pickle.load(f)

X_df['cluster'] = kmeans.labels_.copy()
data_df['cluster'] = kmeans.labels_.copy()
data_non_normalized_df['cluster'] = kmeans.labels_.copy()


#static occlusion
#excluded_states = [2, 89, 98, 54, 71, 49, 3, 97, 10, 74, 52, 58, 99, 6, 69,
#                9, 94, 83, 85, 47, 20, 27, 64, 19, 22, 81, 92, 5, 35, 16]
#data_df = SO.static_occlusion(data_df, excluded_states)

#dynamic occlusion
#ADD DROPPED
with open("dynamic_drop_indices",'rb') as f:
    drop_indices = pickle.load(f)
data_df = SO.dynamic_occlusion(data_df, 0.3, drop_indices=drop_indices)
unique_stay_ids = data_df.index.get_level_values('stay_id').unique()
trajectories = []

for stay_id in unique_stay_ids:
    states, actions = data_df.loc[stay_id]['cluster'], data_df.loc[stay_id]['action']
    trajectory = []
    for i in range(len(states) - 1):
        trajectory.append((states[i], int(actions[i]), states[i+1] ))
    trajectories.append(T.Trajectory(trajectory))

with open(ambig+"trajectories","wb") as f:
    pickle.dump(trajectories,f)

terminal_states = []

for traj in trajectories:
    terminal_states.append(traj._t[-1][-1])

terminal_states = list(set(terminal_states))

smoothing_value = 1
p_transition = np.zeros((num_clusters, num_clusters, 4)) + smoothing_value
for traj in trajectories:
    for tran in traj._t:
        p_transition[tran[0], tran[2], tran[1]] +=1

p_transition = p_transition/ p_transition.sum(axis = 1)[:, np.newaxis, :]
with open(ambig+"p_transition","wb") as f:
    pickle.dump(p_transition, f)

# Convert states and actions to one-hot encoding
state_encoder = OneHotEncoder(sparse=False, categories= [np.arange(num_clusters)])
action_encoder = OneHotEncoder(sparse=False, categories= [np.arange(4)])    
states_onehot = state_encoder.fit_transform(X_df['cluster'].to_numpy().reshape(-1, 1))
actions_onehot = action_encoder.fit_transform(y_df.to_numpy().reshape(-1, 1))

discount = 0.9
# set up features: we use one feature vector per state (1 hot encoding for each cluster/state)
features = state_encoder.transform(np.arange(num_clusters).reshape(-1, 1))

# choose our parameter initialization strategy:
#   initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
#   we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

# actually do some inverse reinforcement learning
# reward_maxent = maxent_irl(p_transition, features, terminal_states, trajectories, optim, init, eps= 1e-3)

reward_maxent_causal = irl_causal(p_transition, features, terminal_states, trajectories, optim, init, discount,
                                  eps=1e-3, eps_svf=1e-4, eps_lap=1e-4)
with open(ambig+"maxent_reward",'wb') as f:
    pickle.dump(reward_maxent_causal, f)
v = reward_maxent_causal
normalized_reward_maxent_causal = (v - v.min()) / (v.max() - v.min())
#print(normalized_reward_maxent_causal)
cluster_sizes = np.zeros(100)

for i in range(num_clusters):
    cluster_sizes[i] = (len(X_df.loc[X_df['cluster'] == i]))
V, Q = S.value_iteration(p_transition, reward_maxent_causal, discount)
Q = Q.reshape((4, num_clusters))
policy_mce = np.argmax(Q, axis = 0).reshape(-1, )
with open(ambig+"maxent_policy", 'wb') as f:
    pickle.dump(policy_mce,f)
