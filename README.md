# simulated-dbtl


This Github repository contains all functions, models, and scripts to reproduce results in the article 
"Simulated Design-Build-Test-Learn Cycles for Consistent Comparison of Machine Learning Methods in Metabolic Engineering". 


### Required python packages
1. Skimpy (https://github.com/EPFL-LCSB/SKiMpy)
2.  pyTFA (https://github.com/EPFL-LCSB/pytfa)
3.  sklearn (https://scikit-learn.org/stable/)
4.  scikit-optimize (https://scikit-optimize.github.io/stable/)
5. scipy (https://scipy.org/)
6. Itertools

### Scripts (~/scripts/ directory): 

.PY SCRIPTS

1. 1210222_combinatorial_space.py: simulates the combinatorial design space of the pathway presented in the paper. This script is required to get all possible combinatorial designs, which is required when you want to calculate the top 100 producers, as a metric. Output of this script is also stored in "data/combinatorial_space_pathwayA.csv. \
--> Enzymes considered: A-G\
--> Enzyme levels (promoters strengths): [0.25,0.5,1,1.5,2,4]

2. comb_scenario_sim_intersect.py: comparison of ML methods for different sampling biases, intersection score of top 100. Results shown in figure 4. Without Bayesian hyperparameter optimization.\
--> Number of designs\
--> scenario:for now three choices: "equal","radical","non-radical".\
--> n_runs: number of runs

Example:

```{r, engine='python', count_lines}
python3 comb_scenario_sim_intersect.py 50 equal 20 
```

3. comb_scenario_sim_r2.py: comparison of ML methods for different training set sizes, r2 value. Results shown in figure 4.\
--> Number of designs\
--> scenario:for now three choices: "equal","radical","non-radical".\
--> n_runs: number of runs


Example:
```{r, engine='python', count_lines}
python3 comb_scenario_sim_r2.py 50 equal 20 
```

4. Bayes_comb_scenario_sim_intersect.py: calculates the intersection of the top 100 prediction, with Bayesian hyperparameter optimization. Three different scenarios are shown in figure 5. For this script, the following things need to be specified through the command lines:\
--> N_designs: the number of strain designs used in the training set.\
--> scenario: for now three choices: "equal","radical","non-radical". These correspond to the training set biases reported in the paper.\
--> n_runs: number of runs\
--> noise_model: "homoschedastic" or "heteroschedastic".\
--> noise_percentage: if not set, will be 0.

Example:
```{r, engine='python', count_lines}
python3 Bayes_comb_scenario_sim_intersect.py 50 equal 20 homoschedastic 0.04
```

5. Bayes_comb_scenario_sim_r2.py: calculates the r2 value, with Bayesian hyperparameter optimization. Results shown in figure 5. Three different scenarios are shown in figure 5. For this script, the following things need to be specified through the command lines:\
--> N_designs: the number of strain designs used in the training set.\
--> scenario: for now three choices: "equal","radical","non-radical". These correspond to the training set biases reported in the paper.\
--> n_runs: number of runs\
--> noise_model: "homoschedastic" or "heteroschedastic".\
--> noise_percentage: if not set, will be 0.

Example:
```{r, engine='python', count_lines}
python3 Bayes_comb_scenario_sim_r2.py 50 equal 20 homoschedastic 0.04
```


6. DBTL_cycle_cost_experiment_1601.py: script to get the results of table 1, DBTL cycles strategies. Note the variable grid, which contains the number of samples used per cycle. \
--> Method: "GradientBoostingRegressor","RandomForestRegressor","SGD_Regressor","MLP_regressor"
--> Strategies: index. In the script, there is a variable "grid" which has a number of predefined sampling strategies. Index wil select grid[:,index] from this list to be simulated. 

Example
```{r, engine='python', count_lines}
python3 DBTL_cycle_cost_experiment_1601.py GradientBoostingRegressor 2
```

.IPYNB SCRIPTS

1. 01_11_comparing scenarios.ipynb: comparison of the different scenario's (equal, biased, and DoE), as shown in figure 5.
2. 01_11_ME_example.ipynb: an example of what a simulated ME scenario looks like, as shown in figure 3.
--> training set size=50 
--> simulation scenario: equally probable sampling of promoters
--> test set size for plot: 1000
3.0311_combspace_analysis.ipynb: analysis of the simulated combinatorial space using the kinetic model (see paper). A development script for the recommendation algorithm described in figure 6. (REM)
4. DBTL_cycle_cost_experiment_1301.ipynb: a development script for the DBTL cycle experiment reported in table 1. (REM)
5. DBTL cycles.ipynb: old development script for the DBTL cycle code. (REM)
6. DBTL cycles V2.ipynb: script used to generate figure S3, S4, S5, S6. Generates additional results for the automated recommendation algorithm.
7. DBTL recommendation (Fig 6).ipynb: the automated recommendation system that was used for comparing ML methods over multiple DBTL cycles. Generates the figures as shown in Figure 6.
8. DBTL strategy (Fig 6).ipynb: not further used (REM)
9. Feature_Importance_Dev.ipynb: development for feature importance in supporting information (REM)
10.noise_experiments.ipynb: test script for noise experiments (supporting information)
11. Noise (Table SI).ipynb: visualization of the results of the noise experiments.
12. Scenarios (Fig 5).ipynb: visualization of the performance of the different scenarios, as shown in figure 5.
13. training set (Fig 4).ipynb: visualization of the performance of the 7 tested algorithms  for different training set sizes, as shown in figure 4.
14. Bayesian test.ipynb: dev script for setting up Bayesian hyperparameter optimization for each algorithm. 
15. figure 2C Biomass modelling.ipynb: script for the batch bioprocess as shown in figure 2C.
16. Local_Stability_PathwayA.ipynb: script for generating the simulation of local perturbations as presented in figure 2A, and 2B.
17. DoE_revision.ipynb: script for comparing the D-optimal design, Latin Hypercube, and Random Sampling. This is reported in the Supporting Information.


### Scripts required for reproducing the main results in the paper:

1. 1210222_combinatorial_space.py: top 100 metric
2. Fig 2: figure 2C Biomass modelling.ipynb, Local_Stability_Pathway.ipynb
3.  Fig 3: 01_11_ME_example.ipynb
4.  Fig 4: training set(Fig 4).ipynb,comb_scenario_sim_intersect.py,comb_scenario_sim_r2.py
5.  Fig 5: Scenarios (Fig 5).ipynb,Bayes_comb_scenario_sim_r2.py, Bayes_comb_scenario_sim_intersect.py
6. Fig 6: DBTL recommendation (Fig 6).ipynb
7. Table 1: DBTL_cycle_cost_experiment_1601.py


### References




