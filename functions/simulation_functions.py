#load packages
from pytfa.io.json import load_json_model
from skimpy.io.yaml import  load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations, load_fluxes
from skimpy.core.parameters import ParameterValues
from skimpy.utils.namespace import *
import pandas as pd
import numpy as np
import skimpy
import time
import matplotlib.pyplot as plt
import itertools
import matplotlib
import sys

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

def setup_ode_system(kmodel,tmodel,ref_solution):    
    # Units of the parameters are muM and hr
    CONCENTRATION_SCALING = 1e6  
    TIME_SCALING = 1 # 1hr to 1min
    DENSITY = 1200 # g/L
    GDW_GWW_RATIO = 0.3 # Assumes 70% Water
    
    #load relevant input files
    kmodel =  load_yaml_model(kmodel)
    tmodel = load_json_model(tmodel)
    ref_solution=pd.read_csv(ref_solution,index_col=0).loc['strain_1',:]
    
    # concentration of the metabolites |
    ref_concentrations = load_concentrations(ref_solution, tmodel, kmodel,
                                         concentration_scaling=CONCENTRATION_SCALING)
    #To run dynamic simulations, the model needs to contain compiled ODE expressions
    # help(kmodel.prepare()): Model preparation for different analysis types. The preparation is done
    #before the compling step
    parameter_values = {p.symbol:p.value for p in kmodel.parameters.values()}
    parameter_values = ParameterValues(parameter_values, kmodel)
    kmodel.prepare()
    kmodel.compile_ode(sim_type=QSSA, ncpu=8)
    
    for k in kmodel.initial_conditions:
        kmodel.initial_conditions[k] = ref_concentrations[k]
    
    return kmodel, ref_concentrations,tmodel
#print(b-a) #26 seconds

def setup_batch_reactor(filenames):
    #Sets up the symbolic batch reactor

    reactor=make_batch_reactor(filenames['batch_file'])
    reactor.compile_ode(add_dilution=False)# check later what dilution does, it says dilution of intracellular metabolites
    tmodel=load_json_model(filenames['tmodel'])
    bkmodel=load_yaml_model(filenames['batch_kmodel'])
    reference_solutions=pd.read_csv(filenames['ref_solution'],index_col=0)
    ref_concentrations= load_concentrations(reference_solutions.loc['strain_1'], tmodel, bkmodel,
                                                      concentration_scaling=reactor.concentration_scaling)
    reactor.initial_conditions['biomass_strain_1'] = 0.1e12 # Number of cells
    reactor.initialize(ref_concentrations, 'strain_1')
    reactor.initialize(ref_concentrations, 'strain_2')
    return reactor, reference_solutions


def ode_integration(kmodel,met_plots,plotting=True):
    #integrates the ODE, if plotting is true will plot the metabolite of interest
    sol=kmodel.solve_ode(np.linspace(0,25,4000),solver_type="cvode") 
    if plotting==True:
        time=sol.time
        concentrations=sol.concentrations

        for i in met_plots:
            plt.plot(time,concentrations[i],label=i)
            plt.xlabel("Time (in s)")
            plt.ylabel("Concentration")
            plt.legend()
        plt.show()
    else:
        pass
    return sol

def perturb_kmodel(kmodel, enz_dict_perturb,parameter_values):
    ## Perturb the model given the design


    #this function works for both the kinetic model and reactor object
    n=len(enz_dict_perturb.keys())
    
    #this is required because python passes mutable objects
    kmodel.parameters = parameter_values 
    
    #perturb model: if only 1 value, or more perturbations
    perturbed_kmodel=kmodel #not overwriting the wt_model 
    if n==1:
        enz_label=list(enz_dict_perturb.keys())[0]
        enz_level=list(enz_dict_perturb.values())[0]
        perturbed_kmodel.parameters[enz_label].value=perturbed_kmodel.parameters[enz_label].value*enz_level
    else:
        enz_label=list(enz_dict_perturb.keys())
        enz_level=list(enz_dict_perturb.values())
        for i,k in enumerate(enz_label):
            perturbed_kmodel.parameters[k].value=perturbed_kmodel.parameters[k].value*enz_level[i]
    return perturbed_kmodel


def generate_perturbation_scheme_reactor(enz_names,perturbation_range):
    #Unused function, batch reactor class is quite slow
    temp=[]
    for i in enz_names:
        temp2="strain_1_"+i #add strain name
        temp.append(temp2)
    enz_names=temp
    ndim=len(enz_names)
    dim_range_list=[]
    for i in range(ndim):
        dim_range_list.append(np.array(perturbation_range))
    #Cartesian product
    cart=[]
    for element in itertools.product(*dim_range_list):
        cart.append(element)
    design_list=[]
    for i in range(len(cart)):
        values=tuple(cart[i])
        design=dict(zip(enz_names,values))
        design_list.append(design)
    return design_list,cart

def generate_perturbation_scheme(enz_names,perturbation_range): #also here make sure
    """Given the perturbation profile,create a dictionary
    Input:
    perturbation_range: perturbation value of vmax w.r.t. wildtype
    ndim: the number of fluxes to consider
    
    Output of dictionaries
    """
    ndim=len(enz_names)
    dim_range_list=[]
    for i in range(ndim):
        dim_range_list.append(np.array(perturbation_range))
    #Cartesian product
    cart=[]
    for element in itertools.product(*dim_range_list):
        cart.append(element)
    design_list=[]
    for i in range(len(cart)):
        values=tuple(cart[i])
        design=dict(zip(enz_names,values))
        design_list.append(design)
    return design_list,cart


def relative_met_change(sol_wt,sol_ps):
    difference=sol_wt.species-sol_ps.species
    difference=np.mean(difference[-5:-1,:],0)
    wt_avg=np.mean(sol_wt.species[-5:-1,:],0)
    x=difference/wt_avg
    rel_decrease=(1-x)
    return rel_decrease

def relative_flux_change(flux_wt,flux_mt):
    flux_w=np.array(list(flux_wt.values()))
    flux_m=np.array(list(flux_mt.values()))
    difference=flux_w-flux_m
    x=difference/flux_w
    rel_decrease=(1-x)
    
    return rel_decrease

def plot_energy_landscape(rel_change,metabolite,enzymes,enz_ind):
    coordinate_list=[]
    #find the coordinates (number of enzymes), and put them in a list
    for i in range(np.shape(rel_change[metabolite])[0]):
        coordinates=rel_change[metabolite].index[i]
        coordinate_list.append(coordinates)
    number_of_dimensions=len(coordinate_list[0])
    unique_heatmap_dims=[0]*number_of_dimensions
    for i in range(number_of_dimensions):
        dim_list=[]
        for k in range(len(coordinate_list)):
            ith_dim_coord=coordinate_list[k][i]
            dim_list.append(ith_dim_coord)
        dim_list=np.unique(dim_list)
        unique_heatmap_dims[i]=dim_list
    #print(unique_heatmap_dims)     
    matrix=np.zeros((len(unique_heatmap_dims[0]),len(unique_heatmap_dims[0])))
    matrix=pd.DataFrame(matrix, index=list(unique_heatmap_dims[enz_ind[0]]) ,
                                           columns=list(unique_heatmap_dims[enz_ind[1]]))
    #now for the coordinates, find the coordinate of the metabolite of interest index i
    #fill in the matrix based on the x_coordinate
    for i in range(np.shape(rel_change[metabolite])[0]):
        coordinates=rel_change[metabolite].index[i]
        #x_coordinate: vmax_forward_pfk will become the rows
        #ycoordinate: vmax_forward_LDH_d will become the columns
        x_coord=np.where(coordinates[enz_ind[0]]==matrix.index)[0] 
        y_coord=np.where(coordinates[enz_ind[1]]==matrix.columns)[0]
        temp_mat=np.array(matrix)
        #fill in the values of the list 
        temp_mat[x_coord,y_coord]=rel_change[metabolite].values[i]
        matrix=pd.DataFrame(temp_mat, index=list(unique_heatmap_dims[enz_ind[0]]) ,
                                           columns=list(unique_heatmap_dims[enz_ind[1]])) 
    #Apparently, here it goes wrong
    fig,ax=plt.subplots()
    im = ax.imshow(matrix.T,cmap='Reds')
    fig.colorbar(im)
    ax.set_xticks(np.arange(np.shape(matrix)[0]))
    ax.set_xticklabels(np.array(matrix.index))
    ax.set_yticks(np.arange(np.shape(matrix)[0]))
    ax.set_yticklabels(np.array(matrix.columns))
    ax.invert_yaxis()
    plt.xlabel(enzymes[enz_ind[0]])
    plt.ylabel(enzymes[enz_ind[1]])
    plt.xticks(rotation=-45)
    plt.title("Relative flux change in "+metabolite)
    plt.legend()
    plt.show()
    return matrix,fig


def energy_function(rel_change,metabolite_names,weights):
    # A linear weighted function for engineering purposes
    # Input:
    # - weights for the metabolites of interest
    #-relchange matrix
    # Output:
    # -energy landscape for the metabolites of interest (later perhaps also biomass, or whatever)
    norm=np.sum(weights)
    weights=weights/norm
    row_names=rel_change.index
    met_of_interest=[]
    indices=list(rel_change.columns)
    for i in range(len(indices)):
        if indices[i] in metabolite_names:
            met_of_interest.append(i)
    rel_change=rel_change.iloc[:, met_of_interest] 
    for i in  range(np.shape(rel_change)[1]):
        x=list(rel_change.iloc[:,i]*weights[i])
        rel_change.iloc[:,i]=x
    energy=np.sum(rel_change,1)
    return energy


def find_max_coord(relative_met_change,metab_of_int,designs):
    #finds the coordinate of the maximum, which are the perturbed 
    
    #input: relative metabolite change compared to wildtype
    #input: metabolite of interest, or a weighted combination of multiple metabolites (depending on objective)
    
    #output: coordinates of the maximum (values of perturbation)
    #the index of 
    x=relative_met_change[metab_of_int]
    value=np.where(x==np.max(x))[0]
    maximum=designs[int(value)]
    fold_increase=x[value]
    return maximum,fold_increase

def vmax_finder(parameter_values):
    ### finds the vmax of the reactions along with their value for the initial testing of ML
    # One of the ugliest codes ever written
    symbol=list(parameter_values.keys())
    nonsymbol=[]
    for i in symbol:
        i=str(i)
        nonsymbol.append(i)
    myvmax_values=list(parameter_values.values())
    vmax_dictionary=dict(zip(nonsymbol,myvmax_values))
    the_value=[value for key, value in vmax_dictionary.items() if 'vmax_forward_' in key.lower()]
    the_key=[key for key, value in vmax_dictionary.items() if 'vmax_forward_' in key.lower()] 
    vmax_dictionary=dict(zip(the_key,the_value))
    return vmax_dictionary
