def decision_tree(X,y,max_depth,min_samples_leaf):

    import numpy as np

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.tree import DecisionTreeRegressor

    import pandas as pd

#     X=pd.get_dummies(X)

 

# Make a decision tree (classifier or regression)

    if y.dtype.name in ('float32','float64'):

        estimator = DecisionTreeRegressor(max_depth =max_depth,min_samples_leaf=min_samples_leaf)

    else:

        estimator = DecisionTreeClassifier(max_depth =max_depth,min_samples_leaf=min_samples_leaf)

    estimator.fit(X,y)

    return estimator

def leaf_aggregates(y,leave_id):

    import pandas as pd

    import numpy as np

    leaves_y=pd.DataFrame({'y': y ,'leaf_id':leave_id,'count':1})

    leaf_df =leaves_y.groupby("leaf_id").agg({"y": "mean","count":"count"})

    comparison_y= np.average(y)

    leaf_df["divergence"]=np.abs(leaf_df["y"]-comparison_y)

    leaf_df.reset_index(inplace=True)

    leaf_df = leaf_df.sort_values(by=['divergence'],ascending=False)

    return leaf_df

    leaf_df.plot.scatter("count","divergence")

def get_rules(estimator,X,target_id,leave_id):

    import pandas as pd

    # A matrix with an indicator for sample i whether node j is in its path

    node_indicator = estimator.decision_path(X)

    # Find the first sample that is in that leaf

    for i in range(len(X)):

        if node_indicator[i,target_id]==1:

            sample_id=i

            break

    # feature and threshold for every node

    feature = estimator.tree_.feature

    threshold = estimator.tree_.threshold

 

    # All of the nodes leading to the leaf

    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:

                                        node_indicator.indptr[sample_id + 1]]

    # dataframe of rules

    rules = pd.DataFrame(columns = ['node_id','feature','sign','threshold'])

    for node_id in node_index:

        if leave_id[sample_id] == node_id:

            continue

        if (X.iloc[sample_id, feature[node_id]] <= threshold[node_id]):

            threshold_sign = "<="

        else:

            threshold_sign = ">"

 

        rules = rules.append({'node_id':node_id,'feature':X.columns[feature      [node_id]],'sign':threshold_sign,'threshold':threshold[node_id]}, ignore_index=True)   

    # organize rules

#     rules =rules[rules.sign=='>'].groupby(["feature","sign"]).agg({"threshold":"max"}).append(rules[rules.sign=='<='].groupby(["feature","sign"]).agg({"threshold":"min"}) ).reset_index().sort_values("feature")

    return rules

# for a given depth, returns all of the leaves

def decision_tree_analyzer(X,y,max_depth=5,min_samples_leaf = 1000):

    estimator = decision_tree(X,y,max_depth,min_samples_leaf)

    leave_id = estimator.apply(X)

    leaf_df = leaf_aggregates(y,leave_id)

    target_id = int(leaf_df.loc[leaf_df["divergence"].idxmax()].leaf_id)

    for target_id in leaf_df.leaf_id:

        rules = get_rules(estimator,X,target_id,leave_id)

        print(leaf_df[leaf_df.leaf_id==target_id])

        print(rules)

        print('')

 

# gives the most divergent leaf for every tree from depth 1 to 'deepest'       

def decision_tree_analyzer2(X,y,deepest=5,min_samples_leaf = 1000):

    for max_depth in range(1,deepest):

        estimator = decision_tree(X,y,max_depth,min_samples_leaf)

        leave_id = estimator.apply(X)

        leaf_df = leaf_aggregates(y,leave_id)

        target_id = int(leaf_df.loc[leaf_df["divergence"].idxmax()].leaf_id)

        rules = get_rules(estimator,X,target_id,leave_id)

        print(leaf_df[leaf_df.leaf_id==target_id])

        print(rules)

        print('')

        
def divergence_table(data,y_column,count_column,min_count=1000):

    import numpy as np
    avg_y=np.mean(data[y_column])

    big_var_table=pd.DataFrame(columns=[y_column, count_column, 'divergence', 'value',

           'variable'])

    for col in data.columns.drop(y_column):

        var_table = data.groupby(col).agg({y_column:"mean",count_column:"count"})

        var_table["divergence"]=np.abs(var_table[y_column]-avg_y)

        var_table=var_table[var_table[count_column]>min_count]

        var_table["value"]=var_table.index

        var_table = var_table.reset_index(drop=True)

        var_table = var_table.sort_values(by=['divergence'],ascending=False)

        var_table["variable"]=col

    #     var_table=var_table.rename(index=str, columns={col: "value"})

        big_var_table=big_var_table.append(var_table,ignore_index=True)

    return big_var_table.sort_values(by=['divergence'],ascending=False)

#todo: output the graph of leaves: count vs average
#todo: use out of sample count and average
#todo: prettify rules - smooth out >2.5 as >=3 if this is an int, use = if there is only one option
