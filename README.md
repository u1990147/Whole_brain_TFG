# Whole-brain development from an exact population-level model
This repository contains the code developed for my final degree project focused on extending the neuronal model proposed by Pietras, Clusella and Montbrió to a whole-brain level using the Neuronumba framework.

(Prlar de la part de alzheimer) 

**Requirements**

The code is written in Python and requires the following packages:
- NumPy
- Matplotlib
  
The Neuronumba library: https://github.com/neich/neuronumba

This code uses the Human Connectome Project (HCP) and ADNI dataset

**Code structure**

- Pietras2025.py: Defines the neuronal model, including parameters and differential equations.
- test_Pietras2025.py: Runs simulations, loads structural connectivity, and generates results.
- transitori_G.py: Analyses the optimal warm-up time to discard transient behaviors from the simulation.
- Alzheimer_FC.py
