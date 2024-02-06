
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import itertools
from itertools import combinations, permutations, chain


warnings.filterwarnings("ignore")
import networkx as nx
import matplotlib.pyplot as plt
import json

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete.CPD import TabularCPD

"""
    Input:
        dataset --> type DataFrame
        bn --> type pygmp Bayesian Network
        G --> type networkx graph
        target --> type string
    
"""
def causal_shapley(
        dataset: pd.DataFrame, 
        bn: BayesianNetwork, 
        G, 
        target: str, 
        sample_idx = -1):
    
    # we interested in this data point
    sample_idx = random.randint(0, len(dataset)) if sample_idx == -1 else sample_idx
    features = dataset.columns.tolist()
    # target = 
    features.remove(target)
    # sample_idx = 10 
    columns = dataset.drop(columns=[target]).columns
    M = 100
    estimated_shapley_val = {}
    # this is datapoint we interested in
    x = dataset.iloc[sample_idx]
    
    # check valid combination
    valid_combination = []
    for r in range(1, len(G.nodes)):
        for comb in combinations(set(G.nodes) - {target}, r):
            if is_valid_subset(G, set(comb) | {target}):
                valid_combination.append(set(comb) | {target})

    print(f'Shapley value estimation for data point [{sample_idx}]:')
    print(x)
    print('Target:',target)

    for feature_of_interest in features:
        print(f'\nfeature_of_interest: ',feature_of_interest)    
        j = dataset.drop(columns=[target]).columns.tolist().index(feature_of_interest)    
        n_features = len(dataset.drop(columns=[target]).columns)
        feature_idxs = list(range(n_features))
        feature_idxs.remove(j)

        marginal_contributions = []
        
        # Calculate shapley value
        for m in range(M):
            # in BN, we cant just randomize feature to know the absence of feature,
            # however we have to delete the arcs of the coresponding nodes
            x_j_idx = set(random_subset(feature_idxs) + [j])
            x_idx = set(x_j_idx) - set([j])
            j_cols = set([columns[c] for c in x_j_idx])
            cols = set([columns[c] for c in x_idx])
            j_model_valid = False
            for comb in valid_combination:
                if (comb == j_cols | {target}):
                    j_model_valid = True
                    
            model_valid = False
            for comb in valid_combination:
                if (comb == cols | {target}):
                    model_valid = True
            
            p_j = 0
            p = 0
            
            # calculate marginal contribution
            # set model with j
            # assume all model valid (evaluate later)
            j_model_bn = intervention(bn, G, j_cols | set([target]))
            j_model_infer = VariableElimination(j_model_bn)
            j_evidence = {}
            for j_col in j_cols:
                j_evidence[j_col] = x[j_col]
            j_q = j_model_infer.query(variables=[target], evidence=j_evidence, joint=True)
            j_q_proba = predict_proba(j_q, target=target)
            p_j = j_q_proba[target+'=yes']
            
            # model without 
            model_bn = intervention(bn, G, cols | set([target]))
            model_infer = VariableElimination(model_bn)
            evidence = {}
            for col in cols:
                evidence[col] = x[col]
                
            q = model_infer.query(variables=[target], evidence=evidence, joint=True)
            q_proba = predict_proba(q, target=target)
            p = q_proba[target+'=yes']

            marginal_contribution = p_j - p
            marginal_contributions.append(marginal_contribution)
            
        print(f'Value: {sum(marginal_contributions)/100}')
        
        estimated_shapley_val[feature_of_interest] = sum(marginal_contributions)/100
    print(estimated_shapley_val)
    
    return estimated_shapley_val

def predict_proba(query, target):
    probabilities = query.values
    state_names = query.state_names[target]
    proba = {}
    
    for state, probability in zip(state_names, probabilities):
        proba[target+'='+state] = probability
    return proba


def random_subset(input_list):
    length = len(input_list)
    # Randomly decide the number of elements in the subset
    n = random.randint(0, length)
    # Sample 'n' elements from the list
    subset = random.sample(input_list, n)
    return subset

def intervention(bn, G, subset):
    other = set(G.nodes) - subset
    
    for node in other:
        parents = set(G.predecessors(node))
        parent_included = parents.intersection(subset)
        if len(parent_included) > 0:
            bn = bn.do(node)

        childs = set(G.successors(node))
        child_included = childs.intersection(subset)
        if len(child_included) > 0:
            bn = bn.do(child_included)
            
    return bn

def is_valid_subset(G, subset):
    """
    Checks if a subset of nodes is valid under do-intervention rules.
    Each node in the subset must include all its predecessors from the graph.
    """
    # check whether the input subset is valid (must have a route to y)
    subG = G.subgraph(subset).copy()
    
    if not nx.is_connected(subG.to_undirected()):
        return False

    for node in subset:
        parents = set(G.predecessors(node))
        parents_included = parents.intersection(subset)
        if len(parents_included) != 0:
            if len(parents_included) < len(parents):
                return False
    return True