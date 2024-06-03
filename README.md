# Scope of Your Project
Assess the extent of performance degradation in non-linear SVMs caused by precision loss during the conversion of model parameters from float to posit formats.
Investigate the relationship between various float and posit format configurations and their impact on the inference accuracy of non-linear SVMs.
Develop and evaluate training strategies that incorporate posit constraints aimed at reducing performance losses during inference with posits.


## Project Structure
The project is organized as follows:
- matlab/ contains the matlab scripts used to load datasets and find the optimal solutions of the optimization problems
- datasets/ contains the datasets used for evaluating our solution
- main_lambdas.cpp contains the code used to perform inference starting from the dual optimal solutions  


## Requirements to run the project
- Matlab & Matlab Optimization Toolbox
- cpp_posit library
  
## Requirements to run the project
- Matlab & Matlab Optimization Toolbox
- cpp_posit library
  
## Experiments
- Dynamic Range Adjustment in SVM with Posit Constraints
- Worst-case Bounds Adjustment in SVM with Posit Constraints
- Density-Sensitive Regularization in SVM with Posit Constraints
- Quantile-based Thresholding



