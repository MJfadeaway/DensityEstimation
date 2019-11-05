from __future__ import division
import numpy as np
from legpoly import *
from numpy.polynomial import polynomial as P

# Author: Kejun Tang
# Last Revised: September 03, 2018

def sigmoid(x):
	"""sigmoid function"""
	"""
	inputs:
	-------
	x    a number where need to be evaluated
	
	returns:
	--------
	sigmoid function value at x
	"""
	return 1/(np.exp(-x)+1)



def part_prod(cpfactor, sample_point, num_basis, idx_variable):
	"""intermediate step, partial product for computing gradient"""
	"""
	inputs:
	-------
	cpfactor    factor matrix
	sample_point    a sample
	idx_variable    index
	
	returns:
	--------
	a product except idx_variable CP factor
	"""
	Phi = np.zeros((len(sample_point), num_basis))
	dim = cpfactor.shape[0]
	assert dim == len(sample_point)
	prod_except_one = 1
	for j in range(len(sample_point)):
		Phi[j,:] = construct_phi(num_basis, sample_point[j])
	for jj in range(dim):
		if jj == idx_variable:
			prod_f = 1
		else:
			prod_f = np.sum(cpfactor[jj,:]*Phi[jj,:])
		prod_except_one *= prod_f

	return prod_except_one


def neg_loglikeli(cpfactor_list, sample_points, num_basis):
	"""computing mean negative loglikelihood function on training set"""
	"""
	inputs:
	-------
	cpfactor_list    factor matrix list consist of coefficients, len(cpfactor_list) = CP rank
	sample_points    batch sample points, each row is a sample
	num_basis    the number of basis, each basis is a Legendre polynomial
	
	returns:
	--------
	mean negative loglikelihood function on a batch training samples
	"""
	nll = 0
	num_sample = sample_points.shape[0]
	for k in range(num_sample):
		nll += -eval_cptensor(cpfactor_list, sample_points[k], num_basis)

	return nll/num_sample



def loss_funappr(cpfactor_list, sample_points, y, num_basis):
	"""computing mse loss function on training set for function approximation"""
	"""
	inputs:
	-------
	cpfactor_list    factor matrix list consist of coefficients, len(cpfactor_list) = CP rank
	sample_points    batch sample points, each row is a sample
	y    true function values
	num_basis    the number of basis, each basis is a Legendre polynomial
	
	returns:
	--------
	mse loss function on a batch training samples
	"""
	mse_loss = 0
	num_sample = sample_points.shape[0]
	assert num_sample == len(y)
	for k in range(num_sample):
		mse_loss += .5 * (eval_cptensor(cpfactor_list, sample_points[k], num_basis) - y[k])**2

	return mse_loss/num_sample



def nll_gradient(cpfactor_list, sample_points, num_basis):
	"""computing the gradient of mean log likelihood function"""
	"""
	inputs:
	-------
	cpfactor_list    factor matrix list consist of coefficients, len(cpfactor_list) = CP rank
	sample_points    batch sample points, each row is a sample
	num_basis    the number of basis, each basis is a Legendre polynomial
	
	returns:
	--------
	the gradient of mean log likelihood function with respect to each cpfactor_list
	"""
	gradient_list = []
	cp_rank = len(cpfactor_list)
	num_sample = sample_points.shape[0]
	for idx_rank in range(cp_rank):
		gradient = []
		for idx_variable in range(sample_points.shape[1]):
			temp_vector = np.zeros(num_basis)
			for idx_sample in range(num_sample):
				phi = construct_phi(num_basis, sample_points[idx_sample, idx_variable])
				#temp_vector += -phi/eval_cptensor(cpfactor_list, sample_points[idx_sample, :], num_basis)
				temp_vector += -phi*(1-sigmoid(eval_cptensor(cpfactor_list, sample_points[idx_sample, :], num_basis)))*part_prod(cpfactor_list[idx_rank], sample_points[idx_sample,:], num_basis, idx_variable)
			gradient.append(temp_vector/num_sample)
		gradient_list.append(np.array(gradient))

	return gradient_list



def mse_gradient(cpfactor_list, sample_points, y, num_basis):
	"""computing the gradient of mean square error for supervised learing problem"""
	"""
	inputs:
	-------
	cpfactor_list    factor matrix list consist of coefficients, len(cpfactor_list) = CP rank
	sample_points    batch sample points, each row is a sample
	num_basis    the number of basis, each basis is a Legendre polynomial
	y    true function values, a vector, numpy array

	returns:
	--------
	the gradient of mean square error function with respect to each cpfactor_list
	"""
	gradient_list = []
	cp_rank = len(cpfactor_list)
	num_sample = sample_points.shape[0]
	for idx_rank in range(cp_rank):
		gradient = []
		for idx_variable in range(sample_points.shape[1]):
			temp_vector = np.zeros(num_basis)
			for idx_sample in range(num_sample):
				phi = construct_phi(num_basis, sample_points[idx_sample, idx_variable])
				temp_vector += (eval_cptensor(cpfactor_list, sample_points[idx_sample], num_basis)-y[idx_sample])*part_prod(cpfactor_list[idx_rank], sample_points[idx_sample,:], num_basis, idx_variable)*phi
			gradient.append(temp_vector/num_sample)
		gradient_list.append(np.array(gradient))

	return gradient_list


def nll_Adam(x, dx, config=None):
	"""Adam optimizer for minimize loss function"""
	"""
	inputs:
	-------
	x    current point
	dx    the gradient at current point
	config    parameters for Adam algorithm
	
	returns:
	--------
	next point after an iteration
	
	Reference: D Kinga, JB Adam - International Conference on Learning Representations 2015 && CS231n assignment 2 optim.py
	"""
	if config is None: config = {}
	config.setdefault('learning_rate', 1e-3) # do not change the variable if learning_rate is exsit
	config.setdefault('beta1', 0.9)
	config.setdefault('beta2', 0.999)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('m', np.zeros_like(x))
	config.setdefault('v', np.zeros_like(x))
	config.setdefault('t', 1)

	config['t'] = config['t'] + 1
	config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
	config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dx * dx
	m_unbias = config['m'] / (1 - config['beta1'] ** config['t'])
	v_unbias = config['v'] / (1 - config['beta2'] ** config['t'])
	next_x = x - config['learning_rate'] * m_unbias / (np.sqrt(v_unbias) + config['epsilon'])

	return next_x, config



def normalize_cp(cpfactor_list, num_basis, bnd):
	"""normalization"""
	"""
	inputs:
	-------
	cpfactor_list    factor matrix list consist of coefficients, len(cpfactor_list) = CP rank
	num_basis    the number of basis, each basis is a Legendre polynomial
	bnd    lower bound and upper bound, a list 
	
	returns:
	--------
	a scalar, definite integral 
	"""
	norm_scalar = 0
	dim = cpfactor_list[0].shape[0]
	cp_rank = len(cpfactor_list)
	for idx_rank in range(cp_rank):
		f_int = 1
		cpfactor = cpfactor_list[idx_rank]
		for idx_var in range(dim):
			uni_lbnd = bnd[idx_var, 0]
			uni_ubnd = bnd[idx_var, 1]
			int_poly = np.polynomial.legendre.legint(cpfactor[idx_var])
			uni_int = 0
			for idx_basis in range(num_basis+1):
				int_basis = P.polyint(list(int_poly[idx_basis]*legendre_poly(idx_basis))[::-1]) # polyint ascending order
				uni_int += eval_poly(list(int_basis)[::-1], uni_ubnd) - eval_poly(list(int_basis)[::-1], uni_lbnd) # polyval descending order
			f_int *= uni_int
		norm_scalar += f_int

	return norm_scalar



def normalize_cpcoeff(cpfactor_list, norm_scalar):
	"""normalization for CP factor"""
	"""
	inputs:
	-------
	cpfactor_list    factor matrix list consist of coefficients, len(cpfactor_list) = CP rank
	norm_scalar    the integral for normalization
	
	returns:
	--------
	normalized cpfactor 
	"""
	dim = cpfactor_list[0].shape[0]
	decomp_norm_scalar = np.power(norm_scalar, dim)
	cp_rank = len(cpfactor_list)
	for idx_rank in range(cp_rank):
		cpfactor_list[idx_rank] = cpfactor_list[idx_rank]/decomp_norm_scalar

	return cpfactor_list
