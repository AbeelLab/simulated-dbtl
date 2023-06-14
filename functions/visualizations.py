import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib import colors
import matplotlib.patches as mpatches


def metabolic_engineering_vis_lancet(training_cart,training_fluxes,percentage):
    training_cart=np.array(training_cart).T
    length=len(np.unique(training_cart))
    col_promoters= ['#00468BFF','#ED0000FF','#42B540FF','#0099B4FF','#925E9FFF','#FDAF91FF']
    col_promoters= col_promoters
    cmap = colors.ListedColormap(col_promoters)
    bounds=[0,0.3,0.6,1.1,1.6,2.1,4.1]
    error=np.random.uniform(low=training_fluxes*(1-percentage),high=training_fluxes*(1+percentage))-training_fluxes
    #error=np.random.normal(loc=training_fluxes, scale=percentage)-training_fluxes
    norm = colors.BoundaryNorm(bounds, cmap.N)
    labels=np.unique(training_cart)
    fig1, axs = plt.subplots(figsize=(15,2))
    fig1.tight_layout()
    plt.bar(np.arange(0,np.shape(training_cart)[1],1),training_fluxes,yerr=error,color="grey",width=0.6)
    plt.axhline(y=1,c="black",linestyle="--")
    axs.set_xticks(np.arange(0, np.shape(training_cart)[1], 1))
    axs.set_xticklabels([])
    #axs.set_yticklabels()
    plt.ylabel("Rel. flux")

    #axs.set_yticks(np.arange(0, np.shape(cart)[0], 1))
    
    fig, axs = plt.subplots(figsize=(15,15))
    #Plot 1
    axs= plt.imshow(training_cart,cmap=cmap,norm=norm)
    axs= plt.gca()
    # Major ticks
    axs.set_xticks(np.arange(0, np.shape(training_cart)[1], 1))
    axs.set_yticks(np.arange(0, np.shape(training_cart)[0], 1))
    # Labels for major ticks
    enz=["Reaction A","Reaction B","Reaction C","Reaction D","Reaction E","Reaction F","Reaction G"]
    axs.set_yticklabels(enz)
    axs.set_xticklabels([])
    # Minor ticks
    axs.set_xticks(np.arange(-.5, np.shape(training_cart)[1],1), minor=True)
    axs.set_yticks(np.arange(-.5, np.shape(training_cart)[0], 1), minor=True)
    # Gridlines based on minor ticks
    axs.grid(which='minor', color="w", linestyle='-', linewidth=2)
    one = mpatches.Patch(color=col_promoters[0], label='0.25')
    two= mpatches.Patch(color=col_promoters[1], label='0.5')
    three = mpatches.Patch(color=col_promoters[2], label='1')
    four= mpatches.Patch(color=col_promoters[3], label='1.5')
    five= mpatches.Patch(color=col_promoters[4], label='2')
    six= mpatches.Patch(color=col_promoters[5], label='4')
    plt.legend(handles=[one, two,three,four,five,six],title="Promoter strength",bbox_to_anchor=(1.14, 1.0))
    return fig1,fig, axs
    
    
def metabolic_engineering_vis(training_cart,training_fluxes,percentage):
    training_cart=np.array(training_cart).T
    length=len(np.unique(training_cart))
    col_promoters= ['#BB0103','#DB7A7B','#ADB6B6FF','#0099B4FF','#00468B99','#00468BFF']
    col_promoters= col_promoters
    cmap = colors.ListedColormap(col_promoters)
    bounds=[0,0.3,0.6,1.1,1.6,2.1,4.1]
    error=np.random.uniform(low=training_fluxes*(1-percentage),high=training_fluxes*(1+percentage))-training_fluxes
    #error=np.random.normal(loc=training_fluxes, scale=percentage)-training_fluxes
    norm = colors.BoundaryNorm(bounds, cmap.N)
    labels=np.unique(training_cart)
    fig1, axs = plt.subplots(figsize=(15,2))
    fig1.tight_layout()
    plt.bar(np.arange(0,np.shape(training_cart)[1],1),training_fluxes,yerr=error,color="grey",width=0.6)
    plt.axhline(y=1,c="black",linestyle="--")
    axs.set_xticks(np.arange(0, np.shape(training_cart)[1], 1))
    axs.set_xticklabels([])
    #axs.set_yticklabels()
    plt.ylabel("Rel. flux")

    #axs.set_yticks(np.arange(0, np.shape(cart)[0], 1))
    
    fig, axs = plt.subplots(figsize=(15,15))
    #Plot 1
    axs= plt.imshow(training_cart,cmap=cmap,norm=norm)
    axs= plt.gca()
    # Major ticks
    axs.set_xticks(np.arange(0, np.shape(training_cart)[1], 1))
    axs.set_yticks(np.arange(0, np.shape(training_cart)[0], 1))
    # Labels for major ticks
    enz=["Reaction A","Reaction B","Reaction C","Reaction D","Reaction E","Reaction F","Reaction G"]
    axs.set_yticklabels(enz)
    axs.set_xticklabels([])
    # Minor ticks
    axs.set_xticks(np.arange(-.5, np.shape(training_cart)[1],1), minor=True)
    axs.set_yticks(np.arange(-.5, np.shape(training_cart)[0], 1), minor=True)
    # Gridlines based on minor ticks
    axs.grid(which='minor', color="white", linestyle='-', linewidth=2)
    one = mpatches.Patch(color=col_promoters[0], label='0.25')
    two= mpatches.Patch(color=col_promoters[1], label='0.5')
    three = mpatches.Patch(color=col_promoters[2], label='1')
    four= mpatches.Patch(color=col_promoters[3], label='1.5')
    five= mpatches.Patch(color=col_promoters[4], label='2')
    six= mpatches.Patch(color=col_promoters[5], label='4')
    plt.legend(handles=[one, two,three,four,five,six],title="Promoter strength",bbox_to_anchor=(1.14, 1.0))
    return fig1,fig, axs
