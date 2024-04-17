# Reconfigurations of Cortical Manifold Structure during Reward-Based Motor Learning

This repository contains the code and analysis for our study on the whole-brain basis of reward-based motor learning, focusing on the coordinated activity of multiple neural systems across the cortex and subcortex. We utilized functional MRI to monitor human brain activity during a reward-based motor task and analyzed the patterns of functional connectivity projected onto a low-dimensional manifold space. Our findings offer a unique perspective on how higher-order brain systems interact with the sensorimotor cortex to facilitate learning, highlighting the dynamic changes in functional coupling between the medial prefrontal and sensorimotor cortex across different learning phases.

## Publication

For more detailed insights into our findings, refer to our paper: [Reconfigurations of cortical manifold structure during reward-based motor learning](https://elifesciences.org/reviewed-preprints/91928).

Authors: **Qasem Nick**, Daniel J. Gale, Corson Areshenkoff, Anouk De Brouwer, Joseph Nashed, Jeffrey Wammes, Tianyao Zhu, Randy Flanagan, Jonny Smallwood, **Jason Gallivan**

For further information or queries, feel free to contact us at jasongallivan@gmail.com or qniksefat@gmail.com.

## Study Overview

**Abstract:** Adaptive motor behavior is crucially dependent on the coordinated activity of multiple neural systems distributed across the brain. While the sensorimotor cortex's role in motor learning is well-established, the interaction between higher-order brain systems and the sensorimotor cortex during learning is less understood. In our study, we used functional MRI to examine human brain activity during a reward-based motor task, focusing on how subjects learned to shape their hand trajectories through reinforcement feedback. We projected patterns of cortical and striatal functional connectivity onto a low-dimensional manifold space, observing how regions expanded and contracted along the manifold during learning. Our results highlight the neural changes that underpin reward-based motor learning and identify distinct transitions in the functional coupling of sensorimotor to transmodal cortex when adapting behavior.

**Epochs Defined in the Study:**
- **Rest:** Subject is not performing the task. 297 trs, with the first 3 trs dismissed.
- **Baseline:** Subject performs the task without reward. 219 trs, with the first 3 trs dismissed.
- **Learning:** Subject begins receiving rewards, divided into early and late sections to differentiate the learning period.
  - **Early:** Marks the onset of understanding task changes, with the first 3 trs dismissed (3:219 trs).
  - **Late:** Indicates when subjects correctly perform the task, focusing on the last 216 trs.

## Final Submission

The code used to generate figures from the paper is available in the file `elife_submission.py`.

## Notebooks

The notebooks are in `large-manifold-nbs/` and are organized as follows:

### Networks and Subnetworks
- **Notebook 0:** Exploration of Yeo's 7 and 17 networks, including cortical and subcortical plots.

### Connectivity Matrices
- **Notebook 1:** Presentation of connectivity matrices and the centering process for computing gradients.

### Gradients (Large-Scale Manifolds)
- **Notebook 2:** Gradient values and visualizations, including 3D plots.
  
  ![Gradient Visualization](https://github.com/qniksefat/gradient-notebooks/blob/master/large-manifold-nbs/plots/g-regions.png?raw=true)

### Statistical Analysis
- **Notebook 3:** Statistical analysis of the data.

### Post-hoc Seed Connectivity
- **Notebook 4:** Analysis of post-hoc seed connectivity.
  
  ![Seed Connectivity](https://github.com/qniksefat/gradient-notebooks/blob/master/large-manifold-nbs/plots/fig4.png?raw=true)

### Behaviour (Task Scores)
- **Notebook 5:** Examination of individual differences in performance.

## Project Structure

- **Data Directory:** Contains subdirectories for different data sources and files related to the task.
- **Notebooks Directory:** Houses Jupyter notebooks for data analysis and visualization.

**To run the notebooks:** Navigate to the `notebooks/` directory and launch Jupyter Notebook.
