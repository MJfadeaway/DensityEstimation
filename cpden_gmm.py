import numpy as np
import cpden_solver as cs
from cp_utils import *


# Author: Kejun Tang
# Last Revised: September 04, 2018

def mix_gaussian(mu, sigma_list, weights, num_sample):
	"""generate samples of mixture Gaussian distribution"""
	"""
	inputs:
	-------
	mu    mean list, numpy array
	sigma_list    sigma list
	weights    weights corresponding to each components
	num_sample    the number of samples
	
	returns:
	--------
	samples
	probability density function (pdf) of mixture Gaussian distribution
	"""
	dim = mu.shape[1]
	num_components = mu.shape[0]
	assert (len(weights) == num_components) and (num_components == len(sigma_list))
	data = np.zeros((num_sample, dim))
	for i in range(num_sample):
		idx_component = np.random.choice(num_components, p=weights)
		mean = mu[idx_component]
		cov = sigma_list[idx_component]
		data[i, :] = np.random.multivariate_normal(mean, cov)
	return data


num_train_sample = 4000
dim = 2
weights = [0.1, 0.1, 0.3, 0.2, 0.1, 0.2]
num_components = len(weights)
mu = np.zeros((num_components, dim))
sigma_list = []
for k in range(num_components):
	mean = np.random.randn(1) * [1,2]
	mu[k, :] = mean
	temp_mat = np.random.randn(dim,dim)
	cov = np.matmul(temp_mat.T, temp_mat)
	sigma_list.append(cov)

mix_gaussian_data = mix_gaussian(mu, sigma_list, weights, num_train_sample)
bnd = np.zeros((dim, 2)) # lower bound and upper bound
for kk in range(dim):
	bnd[kk, 0] = np.min(mix_gaussian_data[:, kk])
	bnd[kk, 1] = np.max(mix_gaussian_data[:, kk])

cpfactor = []
cp_rank = 10
num_basis = 5
for k in range(cp_rank):
	factor = np.random.rand(dim, num_basis)
	cpfactor.append(factor)

parameter_dict = {'batch_size': 20, 'lr_decay': 1.0, 'num_epochs': 10, 'print_every': 10}
cpden_pack = cs.Solver(mix_gaussian_data, cpfactor, num_basis, **parameter_dict)
cpden_pack.train()
final_cpfactor = cpden_pack.cpfactor
norm_scalar = normalize_cp(final_cpfactor, num_basis, bnd)
print('-----norma_scalar-----{:.4}'. format(norm_scalar))
f_value = []
for j in range(num_train_sample):
	value = eval_cptensor(final_cpfactor, mix_gaussian_data[j], num_basis)
	f_value.append(value)
print('-----f_value-----:{:.4}'. format(min(f_value)))
print('-----num of f_value < 0-----: {}'. format(np.sum(np.array(f_value)<0)))
norm_cpfactor = normalize_cpcoeff(final_cpfactor, norm_scalar)
norm_scalar1 = normalize_cp(norm_cpfactor, num_basis, bnd)
print('-----norma_scalar after normalization-----{:.4}'. format(norm_scalar1))
