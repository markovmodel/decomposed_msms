# Independent Markov decomposition
This repository contains code to reproduce numerical simulations and visualizations of
Tim Hempel, Mauricio J. del Razo, Christopher T. Lee, Bryn C. Taylor, Rommie E. Amaro, and Frank Noé:
*Independent Markov Decomposition: Toward Modeling Kinetics of Biomolecular Complexes*,
Proc Natl Acad Sci USA 2021, 118 (31), e2105230118. https://doi.org/10.1073/pnas.2105230118.

(Pre-print version: https://www.biorxiv.org/content/10.1101/2021.03.24.436806v1)

## Overview
### Fig1-scaling-multiple-3state-systems.ipynb
Demonstrates combinatorial explosion of global states vs. independent Markov decomposition. Reproduces Fig. 1 in the manuscript.

### Fig3cd-ion-channel-discrete-model.ipynb
Implementation of Hodgkin-Huxley potassium ion channel model and numerical simulation of the dynamics. Reproduces Fig. 3c and d in the manuscript.

### Fig4cd-ion-channel-optimal-partition.ipynb
Implementation of coupled ion channel dimer and numerical simulation of the dynamics at various coupling strenghts. 
Reproduces Fig. 4 c and d in the manuscript.

### Fig5a-syt-decomposition.ipynb
Dependency graph estimation and visualization for MD data of Synaptotagmin-C2A domain. Reproduces Fig. 5a in the manuscript.

### FigS1-approximation-quality-2-2state-systems.ipynb
Assessment of limited sampling effects using a simple toy model. Reproduces Fig. S1 in the supplemental material.

### FigS3-counterexample-Chignolin.ipynb
Counterexample to IMD with small peptide Chignolin (10 amino acids). Reproduces Fig. S3 in the supplemental material.
