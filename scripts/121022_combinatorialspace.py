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

import skimpy
import time
import matplotlib.pyplot as plt
import itertools
import matplotlib
import sys
sys.path.insert(1, 'functions/')


# benchmark functions
import simulation_functions as sf
import scenarios as sc



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


# functions
def scenario_simulation(designs,kmodel,sol_wt,parameter_values,cart):
    """Simulate a scenario based on the design list
    Input: Vmax Design List, Cartesian Design List,kmodel
    
    #cart, the design list for the index of vmax 
    
    Output: 
    - Relative Flux Change w.r.t Wildtype
    - Relative Metabolite Change w.r.t Wildtype
    - Design list for training"""
    #the perturbation integration
    met_plots="glx_c" #this I should remove from the function, no longer used
    rel_list=[]
    rel_flux_list=[]
    vmax_list=[]
    for i in designs:
        pmodel=sf.perturb_kmodel(kmodel,i,parameter_values)
        sol=sf.ode_integration(pmodel,met_plots,plotting=False)
        #vmax dictionary
        perturbed_values = {p.symbol:p.value for p in pmodel.parameters.values()}
        vmax=sf.vmax_finder(perturbed_values)
        vmax_index=list(vmax.keys())
        vmax=list(vmax.values())
        vmax_list.append(vmax)

        if sol.ode_solution.message=='Successful function return.':

            rel_change=sf.relative_met_change(sol_wt,sol)
            rel_list.append(rel_change)

            pmodel_parameters = {p.symbol:p.value for p in kmodel.parameters.values()}
            pmodel_parameters = ParameterValues(parameter_values, pmodel)
            for j,concentrations in sol.concentrations.iterrows():  
                flux_mt=flux_fun(concentrations,parameters=pmodel_parameters)
            rel_flux=sf.relative_flux_change(flux_wt,flux_mt)
            rel_flux_list.append(rel_flux)
        else:
            my_list=np.zeros(49)
            my_flux_list=np.zeros(65)
            my_list[:]=np.NaN
            my_flux_list[:]=np.NaN
            rel_list.append(my_list)
            rel_flux_list.append(my_flux_list)
    

    # Change in metabolite
    rel_change=pd.DataFrame(np.array(rel_list))
    rel_change.columns=sol_wt.names
    rel_change.index=cart

    #Change in flux
    rel_flux_change=pd.DataFrame(np.array(rel_flux_list))
    rel_flux_change.index=cart
    rel_flux_change.columns=list(flux_wt.keys())

    vmax=pd.DataFrame(np.array(vmax_list))
    vmax.columns=vmax_index
    vmax.index=cart
    vmax['Enzyme_G']=rel_flux_change['Enzyme_G']
    
    training_set=pd.DataFrame(np.array(list(vmax.index)),columns=enz_names)
    training_set['Enzyme_G']=list(vmax['Enzyme_G'])
    return rel_change, rel_flux_change, vmax,training_set



filenames={
    "kmodel":"models/shiki_pathway_testmodel.yml",
    "tmodel":"models/shiki_pathway_testmodel_thermodynamic.json",
    "ref_solution":"data/sample06092022.csv",
    "batch_file":"models/single_species.yaml",
    "batch_kmodel":"models/kin_varma.yml"}

#load kinetic model and get parameter values
kmodel,ref_concentrations,tmodel=sf.setup_ode_system(filenames['kmodel'],filenames['tmodel'],filenames['ref_solution'])
parameter_values = {p.symbol:p.value for p in kmodel.parameters.values()}
parameter_values = ParameterValues(parameter_values, kmodel)

#function in the skimpy package for retrieving the fluxes
flux_fun = make_flux_fun(kmodel, QSSA)


met_plots="glx_c"
#kinetic model wildtype ode integration
sol_wt=sf.ode_integration(kmodel,met_plots,plotting=False)
for i,concentrations in sol_wt.concentrations.iterrows():
            flux_wt=flux_fun(concentrations,parameters=parameter_values)


#enzymes to perturb ,'vmax_forward_PFK'
enz_names=["vmax_forward_Enzyme_A","vmax_forward_Enzyme_B","vmax_forward_Enzyme_C","vmax_forward_Enzyme_D",
           "vmax_forward_Enzyme_E","vmax_forward_Enzyme_F","vmax_forward_Enzyme_G"] #'vmax_forward_LDH_D',
perturb_range=[0.25,0.5,1,1.5,2,4]


#perturb_range=[1,1.1,1.2]
#for the kinetic model
designs,cart=sf.generate_perturbation_scheme(enz_names,perturb_range) 
rel_list=[]
rel_flux_list=[]
vmax_list=[]


#test_met,test_flux,test_vmax,test_set_simulation=scenario_simulation(test_set_designs,kmodel,sol_wt,parameter_values,cart_test)
comb_met,comb_flux,comb_vmax, comb_cart=scenario_simulation(designs,kmodel,sol_wt,parameter_values,cart)

#this has a bug

pd.DataFrame(comb_met).to_csv("results/combinatorial_space/comb_met.csv")
pd.DataFrame(comb_flux).to_csv("results/combinatorial_space/comb_flux.csv")
pd.DataFrame(comb_vmax).to_csv("results/combinatorial_space/comb_cart.csv")
