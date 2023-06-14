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





def get_intersection_scores(train_X,train_Y,test_X,test_Y,topX):
    """Calculates all the intersections for the number of runs, with bayesian optimization"""

    # Get all the ML methods
    #svr

    #sgd
    regr_sgd = make_pipeline(StandardScaler(),SGDRegressor(loss="squared_error",max_iter=1000, tol=1e-3))
    regr_sgd.fit(train_X, train_Y)
    predict_sgd=regr_sgd.predict(test_X)

    #rf
    regr_rf = BayesSearchCV(
    RandomForestRegressor(),
    {
        "min_samples_split":(2,3,4,5,6),
        "min_samples_leaf":(2,3,4,5,6),
        "max_depth": (1,2,3,4,5),
    },
    n_iter=15,
    cv=5)
    regr_rf.fit(train_X,train_Y)
    print(regr_rf.best_estimator_)
    predict_rf=regr_rf.predict(test_X)


    #Gradient boosting Regressor
    # log-uniform: understand as search over p = exp(x) by varying x
    regr_GradBoostReg = BayesSearchCV(
    GradientBoostingRegressor(),
    {
        "min_samples_split":(2,3,4,5,6),
        "min_samples_leaf":(2,3,4,5,6),
        "max_depth": (1,2,3,4,5),
        "learning_rate":(0.0001,0.001,0.01,0.1,0.2,0.3),
        #'learning_rate': (0.01,0.2,0.4,0.6),
        #'min_samples_split': (2,3,4)
    },
    n_iter=15,
    cv=5)
    regr_GradBoostReg.fit(train_X, train_Y)
    print(regr_GradBoostReg.best_estimator_)
    predict_GradBoostReg=regr_GradBoostReg.predict(test_X)

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
    regr_NN.fit(train_X,train_Y)
    print(regr_NN.best_estimator_)
    predict_NN=regr_NN.predict(test_X)
    
    top100=np.argsort(test_Y)[::-1][0:topX]
    
    top100_sgd=np.argsort(predict_sgd)[::-1][0:topX]
    top100_rf=np.argsort(predict_rf)[::-1][0:topX]
    top100_gbr=np.argsort(predict_GradBoostReg)[::-1][0:topX]
    top100_nn=np.argsort(predict_NN)[::-1][0:topX]
    
    
    top100_sgd=len(np.intersect1d(top100,top100_sgd))
    top100_rf=len(np.intersect1d(top100,top100_rf))
    top100_gbr=len(np.intersect1d(top100,top100_gbr))
    top100_nn=len(np.intersect1d(top100,top100_nn))
    
    int_scores=[top100_sgd,top100_rf,top100_gbr,top100_nn]
    return int_scores


   
def run_intersection_benchmark(N_runs,N_runs_averaging,topX, number_of_designs,noise_model,noise_percentage):
    """Wrapper for a specific number of designs
    - INput:
    1) N_runs to average: calculate the intersection x times and average
    2) N_runs: number of times the average is calculated
    3) Number of designs: number of designs in the training set
    4) output: dataframe with all the intersections"""

    top100_sgd=[]
    top100_rf=[]
    top100_gbr=[]
    top100_nn=[]

    for i in range(N_runs):   
        print(i)
        int_top100_sgd=[]
        int_top100_rf=[]
        int_top100_gbr=[]
        int_top100_nn=[]
        for i in range(N_runs_averaging):
            N_designs=number_of_designs
            #Define the scenario 
            if scenario=="equal":
                sc1_designs,sc1_cart=sc.scenario1(perturb_range,N_designs,enz_names)
            elif scenario=="radical":
                sc1_designs,sc1_cart=sc.scenario2(perturb_range,N_designs,enz_names)
            elif scenario=="non-radical":
                sc1_designs,sc1_cart=sc.scenario3(perturb_range,N_designs,enz_names)
            #Retrieve the training set instances
            training_scenario1,training_cart=find_set_designs(comb_space,sc1_cart,enz_names) 

            #add noise
            if noise_model=="homoschedastic":
                noise_G=noise.add_homoschedastic_noise(training_scenario1['Enzyme_G'],noise_percentage)
            elif noise_model=="heteroschedastic":
                noise_G=noise.add_heteroschedastic_noise(training_scenario1['Enzyme_G'],noise_percentage)

            training_scenario1['Enzyme_G']=noise_G

            #Get the test set instances (this function is re-used from the script 
            #where the combinatorial space is not available and therefoer requires finding the instances in the comb
            #space again REWRITE)
            #test_cart=test_unseen_designs(cart,enz_names,4000)
            #test_scenario1,test_cart=find_set_designs(comb_space,test_cart,enz_names) 

            train_X=training_scenario1[enz_names]
            train_Y=training_scenario1['Enzyme_G']
            test_X=comb_space[enz_names]
            test_Y=comb_space['Enzyme_G']

            #Get the top 100
            top100=np.argsort(comb_space['Enzyme_G'])[::-1][0:topX]
            intersection_scores=get_intersection_scores(train_X,train_Y,test_X,test_Y,topX)
            int_top100_sgd.append(intersection_scores[0])
            int_top100_rf.append(intersection_scores[1])
            int_top100_gbr.append(intersection_scores[2])
            int_top100_nn.append(intersection_scores[3])

        int_top100_sgd=np.mean(int_top100_sgd)
        int_top100_rf=np.mean(int_top100_rf)
        int_top100_gbr=np.mean(int_top100_gbr)
        int_top100_nn=np.mean(int_top100_nn)


        top100_sgd.append(int_top100_sgd)
        top100_rf.append(int_top100_rf)
        top100_gbr.append(int_top100_gbr)
        top100_nn.append(int_top100_nn)
    results_intersection1000={"SGD":top100_sgd,"RF":top100_rf,
                            "GBR":top100_gbr,"NN":top100_nn}
    results_intersection1000=pd.DataFrame(results_intersection1000)
    return results_intersection1000

def test_unseen_designs(cart,enz_names,test_set_size):
    """testing predictions from """
    index_set=np.arange(0,len(cart),1)
    random_choice=np.random.choice(index_set,test_set_size,replace=False)
    test_cart=[cart[i] for i in random_choice]
    return test_cart


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



N_runs_averaging=10
topX=100
a=time.time()


results=run_intersection_benchmark(N_runs,N_runs_averaging,topX, N_designs,noise_model,noise_percentage)
b=time.time()
print(b-a)
results=pd.DataFrame(results)

print(results)

results.to_csv("../results/noise_scenario_analysis/Bayes_sc_"+scenario+"_"+noise_model+"intersect_"+str(noise_percentage)+"_"+str(N_designs)+".csv")