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
    
    #svr
    regr_svr = svm.SVR()
    regr_svr.fit(train_x, train_y)
    prediction_svr=regr_svr.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,prediction_svr)
    score_svr=r_value**2
    
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
    regr_rf = RandomForestRegressor()
    regr_rf.fit(train_x, train_y)
    predict_rf=regr_rf.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_rf)
    score_rf=r_value**2

    #print("Random Forest: "+str(regr_rf.score(test_x,test_y)))
    #Elastic Net
    regr_elasticnet=ElasticNet()
    regr_elasticnet.fit(train_x,train_y)
    predict_elasticnet=regr_elasticnet.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_elasticnet)
    score_elasticnet=r_value**2
    
    #print("Elastic Net: "+str(regr_elasticnet.score(test_x,test_y)))
    #KNN
    regr_knn=KNeighborsRegressor(n_neighbors=5,weights="distance")
    regr_knn.fit(train_x,train_y)
    predict_knn=regr_knn.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_knn)
    score_knn=r_value**2
    
    #print("k Nearest Neighbors: "+str(regr_knn.score(test_x,test_y)))
    #GPR
    #regr_GPR=GaussianProcessRegressor()
    #regr_GPR.fit(train_x,train_y)
    #predict_GPR=regr_GPR.predict(test_x)
    #slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_GPR)
    #score_GPR=r_value**2
    
    #print("Gaussian Process Regressor: "+str(regr_GPR.score(test_x,test_y))) 
    #this is probably an algorithm that needs to be finetuned more
    #Gradient boosting Regressor
    regr_GradBoostReg=GradientBoostingRegressor()
    regr_GradBoostReg.fit(train_x,train_y)
    predict_GradBoostReg=regr_GradBoostReg.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_GradBoostReg)
    score_GradBoost=r_value**2
    
    #print("Gradient Boosting Regressor: "+str(regr_GradBoostReg.score(test_x,test_y))) 
    # Neural Network
    regr_NN=MLPRegressor(max_iter=8000,activation="relu", learning_rate="adaptive")
    regr_NN.fit(train_x,train_y)
    predict_NN=regr_NN.predict(test_x)
    slope, intercept, r_value, p_value, std_err = linregress(test_y,predict_NN)
    score_NN=r_value**2
    
    #print("Neural Network: "+str(regr_NN.score(test_x,test_y))) 
    algorithms=["SVR","SGD","RF","EN","kNN","GBR","NN"]
    scores=[score_svr,score_sgd,score_rf,score_elasticnet,score_knn,score_GradBoost,score_NN]
    scores=dict(zip(algorithms,scores))
    return scores 



def find_set_designs(comb_space,tcart,enz_names):
    """finds the training or test set designs in the combinatorial space
    Number of features has to be given
    - combinatorial space
    - cart of either the training scenario or the test set
    - enzyme names"""
    temp=0
    tset = pd.DataFrame()
    for design in tcart:
        sub=comb_space
        sub=sub.loc[sub['vmax_forward_Enzyme_A']==design[0]]
        sub=sub.loc[sub['vmax_forward_Enzyme_B']==design[1]]
        sub=sub.loc[sub['vmax_forward_Enzyme_C']==design[2]]
        sub=sub.loc[sub['vmax_forward_Enzyme_D']==design[3]]
        sub=sub.loc[sub['vmax_forward_Enzyme_E']==design[4]]
        sub=sub.loc[sub['vmax_forward_Enzyme_F']==design[5]]
        sub=sub.loc[sub['vmax_forward_Enzyme_G']==design[6]]
        tset=pd.concat([tset,sub])
    return tset,tcart


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
    
    #add noise
    if noise_model=="homoschedastic":
        noise_G=noise.add_homoschedastic_noise(training_scenario1['Enzyme_G'],noise_percentage)
    elif noise_model=="heteroschedastic":
        noise_G=noise.add_heteroschedastic_noise(training_scenario1['Enzyme_G'],noise_percentage)

    training_scenario1['Enzyme_G']=noise_G


    c=time.time()
    #Get the test set instances (this function is re-used from the script 
    #where the combinatorial space is not available and therefoer requires finding the instances in the comb
    #space again REWRITE)
    
    #test_cart=test_unseen_designs(cart,enz_names,4000)
    #test_scenario1,test_cart=find_set_designs(comb_space,test_cart,enz_names) 
    
    train_x=training_scenario1[enz_names]
    train_y=training_scenario1['Enzyme_G']
    test_x=comb_space[enz_names]
    test_y=comb_space['Enzyme_G']
    
 
    scores=ML_methods(train_x,train_y,test_x,test_y)
    results.append(scores)
    
    
    print(c-a)


results=pd.DataFrame(results)
results.to_csv("../results/noise_scenario_analysis/sc_"+scenario+"_"+noise_model+"_"+str(noise_percentage)+"_R2_"+str(N_designs)+".csv")
