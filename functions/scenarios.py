
import regex as re
import pandas as pd
import numpy as np
import random
from collections import Counter
import pyDOE2
def scenario1(perturb_range,N,enz_names):
    cart=[]
    designs_list=[]
    for i in range(N):
        x=np.random.choice(perturb_range,len(enz_names))
        x=tuple(x)
        cart.append(x)
    for i in range(len(cart)):
        design=dict(zip(enz_names,cart[i]))
        designs_list.append(design)
    return designs_list,cart

def scenario2(perturb_range,N,enz_names):
    #Choose with a certain probability distribution
    cart=[]
    designs_list=[]
    for i in range(N):
        x=np.random.choice(perturb_range,len(enz_names),p=[0.25,0.15,0.1,0.1,0.15,0.25])
        x=tuple(x)
        cart.append(x)
    for i in range(len(cart)):
        design=dict(zip(enz_names,cart[i]))
        designs_list.append(design)
    return designs_list,cart


def scenario3(perturb_range,N,enz_names):
    #Choose with a certain probability distribution
    cart=[]
    designs_list=[]
    for i in range(N):
        x=np.random.choice(perturb_range,len(enz_names),p=[0.1,0.15,0.25,0.25,0.15,0.1])
        x=tuple(x)
        cart.append(x)
    for i in range(len(cart)):
        design=dict(zip(enz_names,cart[i]))
        designs_list.append(design)
    return designs_list,cart

def scenario4(levels,reduction):
    #A fractional factorial approach
    scenario4=pyDOE2.gsd(levels,reduction)

    sc4=np.zeros((np.shape(scenario4)[0],np.shape(scenario4)[1]))
    dictionary_for_sc4={0:1, 1:0.5 ,2:1.5 ,3:2}
    for i in range(np.shape(scenario4)[0]):
        for j in range(np.shape(scenario4)[1]):
            sc4[i,j]=dictionary_for_sc4[scenario4[i,j]]
    enz_names=["vmax_forward_Enzyme_A","vmax_forward_Enzyme_B","vmax_forward_Enzyme_C","vmax_forward_Enzyme_D",
               "vmax_forward_Enzyme_E","vmax_forward_Enzyme_F","vmax_forward_Enzyme_G"]
    my_designs=[]
    cart=[]
    for i in range(len(sc4)):
        my_designs.append(dict(zip(enz_names,sc4[i])))
        cart.append(tuple(sc4[i]))
    return my_designs,cart  
    
def manual_scenario(perturb_range,N,pd_library,enz_names):
    """Given some probability distribution, sample a new list of designs
    Input: 
    Probability distribution matrix: each row is a perturbation value (promoter strength), each column is a enzyme"""
    cart=[]
    designs_list=[]
    for i in range(N):
        x=[]
        for k in range(len(enz_names)):
            library_component=np.random.choice(perturb_range,1,p=pd_library[:,k])
            library_component=float(library_component)
            x.append(library_component)
        x=tuple(x)
        cart.append(x)
    for i in range(len(cart)):
        design=dict(zip(enz_names,cart[i]))
        designs_list.append(design)
    return designs_list,cart

colors = ['#00468BFF','#ED0000FF','#42B540FF','#0099B4FF','#925E9FFF','#FDAF91FF']
def plot_promoter_distribution(enz_names,cart):
    #change names for plotting
    enzymes=[]
    a_dict={}
    for i in enz_names:
        enz_names=i.replace("vmax_forward_","")
        enz_names=enz_names.replace("_"," ")
        enzymes.append(enz_names)
    cart=np.array(cart)
    for j in range(np.shape(cart)[1]):
        x=cart[:,j]
        counts=dict(Counter(x))
        a_dict[enzymes[j]]=counts
    #pd.DataFrame(a_dict).T.plot(kind="bar",stacked=True).legend(bbox_to_anchor=(1.0, 1.0)) 
    a_dict=pd.DataFrame(a_dict).sort_index()
    ax=a_dict.T.plot(kind="bar",stacked="True",color=colors)
    
    ax.legend(bbox_to_anchor=(1.0,1.0),title="Promoter Strength")
    ax.set_ylabel("Number of designs")

    return ax


def add_noise(fluxes,percentage):
    """Adds uniform noise to the observation proportional to the mean
    set flux to zero if it becomes negative"""
    error=np.random.uniform(low=fluxes*(1-percentage),high=fluxes*(1+percentage))-fluxes
    noised_fluxes=fluxes+error
    noised_fluxes[noised_fluxes<0]=0
    return noised_fluxes
    


