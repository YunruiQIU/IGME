IGME
==========

This code is used to generate Integrative-Generalized-Master-Equation (IGME) models to analyze state-based dynamics.

IGME is an algorithm designed for analysing data of Molecular Dynamics (MD) Simulation based on Generalized
Master Equation (GME) Theory. Some references are recommended: J. Chem. Phys.,153, 014105, (2020)

Using IGME algorithm, the dynamics projected to a subspace from phase space can be predicted accurately. IGME algorithm takes advantages of the integration method which can effectively remove the fluctuations from the MD data.

Workflow
--------

An example workflow might be as follows:

1. Set up a system for molecular dynamics, and run one or more simulations
   for as long as you can on as many CPUs or GPUs as you have access to.
   There are a lot of great software packages for running MD, e.g
   [OpenMM](https://simtk.org/home/openmm), [Gromacs](http://www.gromacs.org/),
   [Amber](http://ambermd.org/), [CHARMM](http://www.charmm.org/), and
   many others.

2. Transform your MD coordinates into an appropriate set of features.

3. Perform some sort of dimensionality reduction with tICA, SRVnet or VAMPnet.
   Reduce your data into discrete states by using clustering or splitting-lumping.

4. Fit an MSM, (usually the number of states should exceed 50 for IGME calculation).

5. Use Transition Probability Matrix (TPM) computed from MSM at different lag time to be
the input of the IGME. IGME can accurately predict the TPM at any time ,thus gives accurate
estimation for long-time dynamics. The output of IGME is two matrixes, A matrix and T matrix. 
T matrix contains information for a infinite long time behavior and can be used to calculate 
some properties such as MFPT and stationary population.
