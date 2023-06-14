from pytfa.io.json import load_json_model
from skimpy.io.yaml import  load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations, load_fluxes
from skimpy.core.parameters import ParameterValues
from skimpy.utils.namespace import *
from skimpy.core.modifiers import *
from skimpy.io.yaml import load_yaml_model
from skimpy.core.reactor import Reactor
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations, load_fluxes
from skimpy.viz.plotting import timetrace_plot
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations
from skimpy.core.parameters import load_parameter_population
from skimpy.simulations.reactor import make_batch_reactor
from skimpy.core.solution import ODESolutionPopulation
from skimpy.utils.namespace import *
from skimpy.viz.escher import animate_fluxes, plot_fluxes
import copy
from skimpy.io.yaml import export_to_yaml
from skimpy.analysis.ode.utils import make_flux_fun

import pandas as pd
import numpy as np

#import seaborn as sns
import skimpy
import time
import matplotlib.pyplot as plt
import itertools
import matplotlib
import sys
sys.path.insert(1, '../functions/')

# benchmark functions
import simulation_functions as sf
import scenarios as sc
import visualizations as vis
import noise as noise

#ML methods
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import  AdaBoostRegressor
from scipy.stats import linregress

from skopt import BayesSearchCV

N_designs=int(sys.argv[1])
scenario=str(sys.argv[2])
N_runs=int(sys.argv[3])
noise_model=str(sys.argv[4])
noise_percentage=float(sys.argv[5])



print("N_designs: ", N_designs)
print("scenario:", scenario)
print("N_runs:", N_runs)
print("noise_model:",noise_model)
print("noise percentage:",noise_percentage)

if noise_model==None:
    print("noise model is not existing, set noise to 0")
    noise_percentage=0


def ML_methods(train_x,train_y,test_x,test_y):
    "wrapper s.t. we can call it iteratively for each scenario"
    #train test split
    #X=['vmax_forward_Enzyme_A','vmax_forward_Enzyme_B',"vmax_forward_Enzyme_C",
     #  "vmax_forward_Enzyme_D","vmax_forward_Enzyme_E","vmax_forward_Enzyme_F","vmax_forward_Enzyme_G"]
    #train_x=training_set_sc1[X]
    #train_y=training_set_sc1['Enzyme_G']
    #test_x=test_set_simulation[X]
    #test_y=test_set_simulation['Enzyme_G']
    
    
    #score_svr=regr_svr.score(test_x,test_y)
    #print("Support Vector Regressor: "+str(regr_svr.score(test_x,test_y)))
    #sgd
    regr_sgd = make_pipeline(StandardScaler(),SGDRegressor(loss="squared_error",max_iter=1000, tol=1e-3))
    regr_sgd.fit(train_x, train_y)
    predict_sgd=regr_sgd.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_sgd)
    score_sgd=r_value**2
    

    #print("Stochastic Gradient Descent Regressor: "+str(regr_sgd.score(test_x,test_y)))
    #rf
    regr_rf = BayesSearchCV(
    RandomForestRegressor(),
    {
        "min_samples_split":(2,3,4,5,6,7),
        "min_samples_leaf":(2,3,4,5,6,7),
        "max_depth": (1,2,3,4,5,7),
    },
    n_iter=15,
    cv=5)
    regr_rf.fit(train_x,train_y)
    print(regr_rf.best_estimator_)
    predict_rf=regr_rf.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_rf)
    score_rf=r_value**2

    
    #Gradient boosting Regressor
    # log-uniform: understand as search over p = exp(x) by varying x
    regr_GradBoostReg = BayesSearchCV(
    GradientBoostingRegressor(),
    {
        "min_samples_split":(2,3,4,5,6,7),
        "min_samples_leaf":(2,3,4,5,6,7),
        "max_depth": (1,2,3,4,5,7),
        "learning_rate":(0.0001,0.001,0.01,0.1,0.2,0.3),
        #'learning_rate': (0.01,0.2,0.4,0.6),
        #'min_samples_split': (2,3,4)
    },
    n_iter=15,
    cv=5)
    
    regr_GradBoostReg.fit(train_x, train_y)
    print(regr_GradBoostReg.best_estimator_)
    predict_GradBoostReg=regr_GradBoostReg.predict(test_x)
    
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_GradBoostReg)
    score_GradBoost=r_value**2
    
    #print("Gradient Boosting Regressor: "+str(regr_GradBoostReg.score(test_x,test_y))) 
    # Neural Network
    regr_NN = BayesSearchCV(
    MLPRegressor(),
    {
        "activation":("logistic","tanh","relu"),
        "alpha":(0.0001,0.01),
        "max_iter":(5000,8000),
        "hidden_layer_sizes": (1,20),
        "learning_rate":("invscaling","constant","adaptive"),
    },
    n_iter=15,
    cv=5)
    regr_NN.fit(train_x,train_y)
    print(regr_NN.best_estimator_)
    predict_NN=regr_NN.predict(test_x)
    
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_NN)
    score_NN=r_value**2
    
    #print("Neural Network: "+str(regr_NN.score(test_x,test_y))) 
    algorithms=["SGD","RF","GBR","NN"]
    scores=[score_sgd,score_rf,score_GradBoost,score_NN]
    scores=dict(zip(algorithms,scores))
    return scores 





#Load simulations
comb_space=pd.read_csv("../data/combinatorial_space/combinatorial_space_pathway_A.csv")

#enzyme names and perturbation range
enz_names=["vmax_forward_Enzyme_A","vmax_forward_Enzyme_B","vmax_forward_Enzyme_C","vmax_forward_Enzyme_D",
           "vmax_forward_Enzyme_E","vmax_forward_Enzyme_F","vmax_forward_Enzyme_G"] #'vmax_forward_LDH_D',
perturb_range=[0.25,0.5,1,1.5,2,4]
designs,cart=sf.generate_perturbation_scheme(enz_names,perturb_range)



results=[]
for i in range(N_runs):
    print(i)
    a=time.time()

    #Define the scenario 
    if scenario=="equal":
    	sc1_designs,sc1_cart=sc.scenario1(perturb_range,N_designs,enz_names)
    elif scenario=="radical":
        sc1_designs,sc1_cart=sc.scenario2(perturb_range,N_designs,enz_names)
    elif scenario=="non-radical":
        sc1_designs,sc1_cart=sc.scenario3(perturb_range,N_designs,enz_names)
    b=time.time()

    #Retrieve the training set instances
    training_scenario1,training_cart=find_set_designs(comb_space,sc1_cart,enz_names) 
    c=time.time()
    #Get the test set instances (this function is re-used from the script 
    #add noise

    if noise_model=="homoschedastic":
        noise_G=noise.add_homoschedastic_noise(training_scenario1['Enzyme_G'],noise_percentage)
    elif noise_model=="heteroschedastic":
        noise_G=noise.add_heteroschedastic_noise(training_scenario1['Enzyme_G'],noise_percentage)

    training_scenario1['Enzyme_G']=noise_G
    train_x=training_scenario1[enz_names]
    train_y=training_scenario1['Enzyme_G']
    test_x=comb_space[enz_names]
    test_y=comb_space['Enzyme_G']
    
 
    scores=ML_methods(train_x,train_y,test_x,test_y)
    results.append(scores)
    
    
    print(c-a)



results=pd.DataFrame(results)

results.to_csv("../results/noise_scenario_analysis/Bayes_sc_"+scenario+"_"+noise_model+"_"+str(noise_percentage)+"R2_"+str(N_designs)+".csv")
