# Trait-Concentration-and-Self-Organization
Files and code for Similarity Suppresses Cyclicity: Why Similar Competitors Form Hierarchies

# Subfolders
* Figures: Figures generated in the project
* Core Files: Base functions/etc. required for most functions, like performance functions and Hessians/gradients
* Moran: Files for Moran process used to generate performance function for bimatrix games
* Data Files: MATLAB data files generated from tests
* Softmax: New files added for Softmax reproduction

# To generate Figure 1 in the SIAP paper
* Have example_performance_6.m and Hessian_for_example_performance_6.m from "Core Files" folder in the active directory (performance function and Hessian for the performance)
* Have PlotResults_RPFSM.m in the "Softmax" folder in the active directory
* Run Evolution_Test_RPF_softmax.m in the "Softmax" folder with softmax_parameter set to 0 (non-softmax selection parameter), and other parameters set to control (num_competitors = 250, num_traits = 4, num_frequencies = 2, f_mode = 3, genetic_drift = 1`*` 10^(-3), games_per_competitor = 100). Alter the trig_amplitude and linear_amplitude parameters for each line in the graph.

# To generate Moran figures
* Have Hessian_for_performance_chicken_moran_24_individuals.m, Hessian_for_performance_pd_moran_24_individuals.m, Hessian_for_performance_stag_moran_24_individuals.m, performance_pd_moran_24_individuals, performance_stag_moran_24_individuals.m from "Core Files" folder in the active directory (performance function and gradient/Hessian)
* Have Moran_performance_function_interp.m from "Moran" folder in the active directory (to provide performance function for chicken)
* Have log_odds_moran_chicken_10000_midway.m from "Data Files/Bimatrix Performance from Moran" in active directory
* Have PlotResults_Moran from root folder in the active directory
* Run either "Evolution_Test_moran_parallel.m" from "Moran" folder or "Evolution_Test_moran_softmax.m" from "Softmax experiment" folder, depending on whether you want to run the softmax experiment. Alter f_mode parameter to change which game is run.
