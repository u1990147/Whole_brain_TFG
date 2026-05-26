# Whole-brain development from an exact population-level model
This repository contains the code developed for my final degree project focused on extending the population model proposed by Pietras, Clusella and Montbrió to a whole-brain level using the Neuronumba library, and apply it to Alzheimer's disease.

**Original Article:** Pietras, B., Clusella, P., & Montbrió, E. (2025). Exact mean-field description of networks of quadratic integrate-and-fire neurons with synaptic dynamics. Physical Review E, 111(1), 014422.
https://doi.org/10.1103/PhysRevE.111.014422

**Requirements**

The code is written in Python and requires the following packages:
- NumPy
- Matplotlib
- Pathos
  
The Neuronumba library: https://github.com/neich/neuronumba

This code uses the Human Connectome Project (HCP) and Alzheimer’s Disease Neuroimaging Initiative (ADNI) datasets.

**Code structure**

- Pietras2025.py: Defines the neuronal model, including parameters and differential equations.
- test_Pietras2025.py: Loads the structural connectivity matrix from the HCP dataset and defines the simulation parameters. Function run_max_firing_rate calculates the average firing rate activity level in the brain for a given coupling strength. run_BOLD generates a simulated BOLD signal.
- transitori_G.py: Analyses the optimal warm-up time to discard transient behaviors from the simulation.
- alzheimer_FC.py: Computes Functional Connectivity (FC) matrices for the subjects selected from the ADNI dataset. For each subject, it performes five independent simulations across a range of G values and computes its FC matrix. Calculates the average of this five FC computed and compares it to the empirical FC of the subject.
- plot_alzheimer_FC.py: Generates the raincloud plot from the data obtained in Alzheimer_FC.py.
- validate_pietras2025.py: Validates that our code Pietras2025.py and Pietras, Clusella and Montbrió’s model generate the same results. 

