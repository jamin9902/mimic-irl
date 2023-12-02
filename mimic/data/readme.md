The baseline data taken directly from mimic is pickled in:

test_df

train_df

valid_df

The action and observation suboptimality datafarmes are pickled in:

act_test_df

act_train_df

act_valid_df

obs_test_df

obs_train_df

obs_valid_df

These three sets dataframes are then normalized before being passed to the KMeans clusterer from SciPy. These clusters are pickled in

clusters

act_clusters

obs_clusters

and the numbers of points in each cluster are pickled in

counts (= static counts = dynamic counts)

act_counts

obs_counts

The baseline dataframe from test_df, train_df, and valid_df are then processed with the additional static occlusion and dynamic occlusion suboptimalities. The corresponding trajectories calculated from the processed dataframes and clusters are pickled in

trajectories

act_trajectories

obs_trajectories

static_trajectories

dynamic_trajectories

and these are used to compute the transition matrices which are pickled in

p_transition

act_p_transition

obs_p_transition

static_p_transition

dynamic_p_transition

and these data can be used as the inputs for BC, MaxEnt, and GAIL.
