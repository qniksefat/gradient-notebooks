# large-manifold-nbs

This project investigates the whole-brain basis of reward-based motor learning by examining the coordinated activity of multiple neural systems distributed across cortex and subcortex. We used functional MRI to monitor human brain activity during a reward-based motor task and projected patterns of cortical and subcortical functional connectivity onto a low-dimensional manifold space to examine how regions expanded and contracted during learning. Our findings provide a unique characterization of how higher-order brain systems interact with sensorimotor cortex to guide learning and show that transitions across different phases of learning are associated with changes in functional coupling between medial prefrontal and sensorimotor cortex.

## Epoch
Time-periods during the task as follows. Each epoch is set to be 216 time-trials of each ~ 2 seconds. Other time-trials dismissed.
- rest Subject is not doing the task. 297 trs. First 3 trs dismissed.
- baseline Subject is doing the task but no reward is given. 219 trs. First 3 trs dismissed.
- learning Subject starts getting rewards. 619 trs. Divided into early and late sections to differentiate learned period.
  - early When subject starts knowing how the task has changed. First 3 trs dismissed => 3:219 trs.
  - late When some subjects got it right. The last 216 trs.

# Notebooks
## 0 Networks and Subnetworks
In notebook 0, we investigate Yeo's 7 and 17 networks and provide cortical and subcortical plots. The latter can be found in notebook 0.1.

## 1 Connectivity Matrices
Notebook 1 contains the connectivity matrices and shows how we centered them for computing gradients. The effect of centering can be seen in notebook 1.1.

## 2 Gradients (Large-Scale Manifolds)
Notebook 2 includes the gradient values and a few visualizations, as well as 3D plots in notebook 2.1.

![grad1](https://github.com/qniksefat/gradient-notebooks/blob/master/large-manifold-nbs/plots/g-regions.png?raw=true)

## 3 Statistical Analysis
Notebook 3 contains the statistical analysis of the data.

## 4 Post-hoc Seed Connectivity
Notebook 4 provides an analysis of post-hoc seed connectivity.

![seed-conn](https://github.com/qniksefat/gradient-notebooks/blob/master/large-manifold-nbs/plots/fig4.png?raw=true)


## 5 Behaviour (Task Scores)
Notebook 5 revisits the study by looking at individual differences in performance.

# Project Structure and Usage
The data/ directory contains subdirectories for different data sources and files for the task. The notebooks/ directory contains Jupyter notebooks for data analysis and visualization.

To run the notebooks, navigate to the notebooks/ directory and launch Jupyter Notebook.
