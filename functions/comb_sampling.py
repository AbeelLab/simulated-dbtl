

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../functions/')

# benchmark functions
import simulation_functions as sf
import scenarios as sc
import visualizations as vis

from scipy.integrate import simpson
from scipy.stats import entropy



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



def generate_frequency_matrix(comb_space,threshold,enz_names,perturb_range):
    """Given the flux threshold, gives the frequency of DNA promoters in the set above the threshold """
    frequency_matrix=np.zeros((len(enz_names),len(perturb_range)))
    # what are these blobs
    subset=comb_space.loc[comb_space['Enzyme_G']>threshold]
    subset=np.array(subset[enz_names])
    #counts number of remaining designs
    remaining_designs=len(subset)/np.shape(comb_space)[0]
    for i in range(np.shape(subset)[1]):
        for j in range(len(perturb_range)):
            frequency=len(np.where(subset[:,i]==perturb_range[j])[0])
            probability=frequency/np.shape(subset)[0]
            frequency_matrix[i,j]=probability
    frequency_matrix=pd.DataFrame(frequency_matrix,index=enz_names,columns=perturb_range)
    frequency_matrix=frequency_matrix.transpose()
    return frequency_matrix,remaining_designs

def scan_combinatorial_space(comb_space,perturb_range,enz_names,step_size=0.005):
    """"Scans over the combinatorial space and calculates the frequency matrix"""
    largest_value=np.max(comb_space['Enzyme_G'])-0.00001
    threshold_range=np.arange(0.0001,largest_value,step_size)[::-1]
    a3d_freq_mat = np.zeros((len(perturb_range), len(enz_names), len(threshold_range)))
    
    remaining_design_list=[]
    #frequency_matrix=generate_frequency_matrix(comb_space,largest_value)
    for i,k in enumerate(threshold_range):
        frequency_matrix,remaining_designs=generate_frequency_matrix(comb_space,k,enz_names,perturb_range)
        a3d_freq_mat[:,:,i]=frequency_matrix
        remaining_design_list.append(remaining_designs)
    return threshold_range,a3d_freq_mat, remaining_design_list


def get_feature_importance(matrix,comb_space):
    """get the features that contribute most to increasing flux,
    INPUT:
    -the threshold probability matrix:3D
    - the combinatorial sapce"""
    entropies=[]
    enzymes=np.arange(0,np.shape(matrix)[1])
    #loop over the 7 enzymes
    probability_matrix=np.zeros((np.shape(matrix)[0],np.shape(matrix)[1]))
    max_flux=np.max(comb_space['Enzyme_G'])
    for j in range(np.shape(matrix)[1]):
        enz_025=[]
        enz_05=[]
        enz_1=[]
        enz_15=[]
        enz_2=[]
        enz_4=[] 
        # loop over the flux threshold
        for i in range(np.shape(matrix)[2]):
            enz_025.append(matrix[0,enzymes[j],i])
            enz_05.append(matrix[1,enzymes[j],i])
            enz_1.append(matrix[2,enzymes[j],i])
            enz_15.append(matrix[3,enzymes[j],i])
            enz_2.append(matrix[4,enzymes[j],i])
            enz_4.append(matrix[5,enzymes[j],i])
        area_1 = simpson(enz_025, dx=0.005)/max_flux
        area_2 = simpson(enz_05, dx=0.005)/max_flux
        area_3 = simpson(enz_1, dx=0.005)/max_flux
        area_4 = simpson(enz_15, dx=0.005)/max_flux
        area_5 = simpson(enz_2, dx=0.005)/max_flux
        area_6 = simpson(enz_4, dx=0.005)/max_flux
        probability_distribution=[area_1,area_2,area_3,area_4,area_5,area_6]
        probability_distribution=probability_distribution/np.sum(probability_distribution)
        probability_matrix[:,j]=probability_distribution
        x=entropy(probability_distribution)
        entropies.append(x)
    enz_numbers=np.argsort(entropies)
    sorted_entropies=np.sort(entropies)
    return enz_numbers,sorted_entropies, probability_matrix

def feature_entropies(prediction,params,plotting=False):
    """Given the predictive model, use the frequency distribution of promoters as a function of the flux threshold to examine the entropy decrease over time
    "Input:
    the predicted value of the combinatorial design space
    output:
    feature importances"""
    comb_space=params['combinatorial_space']
    pred_comb_space=comb_space.copy()
    pred_comb_space['Enzyme_G']=prediction
    threshold,matrix_pred,remaining_designs_pred=scan_combinatorial_space(pred_comb_space,
                                                                                params['perturb_range'],
                                                                                params['enz_names'],0.005)

    enz_names=params['enz_names']
    entropy_thresholded=np.zeros((np.shape(matrix_pred)[1],np.shape(matrix_pred)[2]))
    for i in range(np.shape(matrix_pred)[2]):
        for j in range(np.shape(matrix_pred)[1]):
            entropy_thresholded[j,i]=entropy(np.transpose(matrix_pred[:,j,i]))

    
    areas=[]
    names=enz_names
    for i in range(len(enz_names)):
        area_X = 1-simpson(entropy_thresholded[i,:], dx=0.005)/(entropy([1/6,1/6,1/6,1/6,1/6,1/6])*np.max(threshold))
        areas.append(area_X)
    feature_importance=dict(zip(names,areas))

    if plotting==True:
        plt.plot(threshold,entropy_thresholded[0,:],label="A")
        plt.plot(threshold,entropy_thresholded[1,:],label="B")
        plt.plot(threshold,entropy_thresholded[2,:],label="C")
        plt.plot(threshold,entropy_thresholded[3,:],label="D")
        plt.plot(threshold,entropy_thresholded[4,:],label="E")
        plt.plot(threshold,entropy_thresholded[5,:],label="F")
        plt.plot(threshold,entropy_thresholded[6,:],label="G")
        plt.xlabel("Flux threshold")
        plt.ylabel("Entropy")
        plt.legend()
        plt.ylim(0,2)
        plt.axhline(1.791759469228055,c="black",linewidth=4,linestyle="--")
        plt.title("Entropy as a function of threshold")
        plt.show()
    if plotting==False:
        pass
    return feature_importance

def reference_noise_model(prediction,params, N_runs):
    """Shuffling y-values such that correlation structures are lost"""
    random_entropy=[]
    comb_space=params['combinatorial_space']
    
    pred_comb_space=comb_space.copy()
    pred_comb_space['Enzyme_G']=prediction

    enz_names=params['enz_names']
    for i in range(N_runs):
        print(i)
        random_comb_space=comb_space.copy()
        random_comb_space['Enzyme_G']=np.random.permutation(random_comb_space['Enzyme_G'])

        threshold,random_matrix_pred,remaining_designs_pred=scan_combinatorial_space(random_comb_space,
                                                                                params['perturb_range'],
                                                                                params['enz_names'],0.005)


        random_entropy_thresholded=np.zeros((np.shape(random_matrix_pred)[1],np.shape(random_matrix_pred)[2]))
        for i in range(np.shape(random_matrix_pred)[2]):
            for j in range(np.shape(random_matrix_pred)[1]):
                random_entropy_thresholded[j,i]=entropy(np.transpose(random_matrix_pred[:,j,i]))

        for i in range(len(enz_names)):
            area_random_entropy =1-simpson(random_entropy_thresholded[i,:], dx=0.005)/(entropy([1/6,1/6,1/6,1/6,1/6,1/6])*np.max(threshold))
            random_entropy.append(area_random_entropy)
        mu_random_entropy=np.mean(random_entropy)
        si_random_entropy=np.std(random_entropy)
    return mu_random_entropy,si_random_entropy


def get_significance_statement(feature_importance,mu_random_entropy,si_random_entropy,std):
    """Gets the significnce w.r.t. the reference noise model"""
    feature_importance_values=list(feature_importance.values())
    feature_importance_keys=list(feature_importance.keys())
    significance=[]
    for i in range(len(feature_importance_values)):
        temp=feature_importance_values[i]>mu_random_entropy+(si_random_entropy*std)
        significance.append(temp)
    significance=dict(zip(feature_importance_keys,significance))
    return significance
