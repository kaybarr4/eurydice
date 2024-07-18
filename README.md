A WIP to create a package that will perform cross-validations for radial velocity signal models with Gaussian processes 💃🕺


Last Updated: July 17th 2024

Current Version: 0.2.2

v.0.2 Updates (July 10th 2024):
* autoformatting
* splitting code into modules: kepler (for keplerian helper functions), plot (to handle plotting), and CV (holding the CrossValidation object + all GPR code). 
* updating some old error messages in kepler.calc_keplerian_signal function
* added a default kernel for the CrossValidation object to use if a predefined one is not passed
* updating plotting functions to return a matplotlib.plyplot.Figure
* fitting Gaussians to the residuals on the histogram plot used for CV
* added functionality for the CrossValidation object to calculate and utilize an N body keplerian mean function 

To-Do:
* adding compatibility with pre-existing GP packages (george, tinygp, celerite): the gist is probably to let code read if the kernel function its passed is a certain object (e.g a george kernel object) and let the code run the GPR and CV using that package's methods
* writing a tutorial on how to use code with juypter notebook
* combing through existing code to improve lacking documentation and find more appropriate places to include error messages
* modifying split function in CrossValidation to let users have a choice as to split data randomly or not (e.g whether to split data 80/20 randomly for training or to split the data such that the first 80 data points chronologically are used to train the model and the last 20 used to assess predictability)
* writing more tests: how to write tests for GP_predict, run_CV and plotting functions? seems too convoluted for my pea brain at the moment 🫨
