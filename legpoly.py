import scipy.special as sp
import numpy as np


# Author: Kejun Tang
# Last Revised: September 02, 2018

def legendre_poly(n, monic=False):
	"""compute a Legendre polynomial"""
	"""
	inputs:
	-------
	n    degree of polynomial
	monic    monic is True if the leading coefficient to be 1

	returns:
	--------
	Legendre polynomial with descending power ranking
	"""
	return sp.legendre(n, monic)



def eval_poly(poly_coeff, x):
	"""evaluating polynomial function at x"""
	"""
	inputs:
	-------
	poly_coeff    the coefficients of polynomial with descending power ranking
	x    a number where need to be evaluating

	returns:
	--------
	polynomial function values at x
	"""
	return np.polyval(poly_coeff, x)



def construct_phi(num_basis, x):
	"""construct Legendre basis vector at sample_point"""
	"""
	inputs:
	-------
	num_basis    the number of basis, each basis is a Legendre polynomial
	x    a number

	returns:
	--------
	a vector with entries determined by Legendre polynomial at x
	"""
	phi = np.zeros(num_basis)
	for i in range(num_basis):
		phi[i] = eval_poly(legendre_poly(i), x)

	return phi



def eval_cptensor(cpfactor_list, sample_point, num_basis):
	"""evaluating function values at sample_point"""
	"""
	inputs:
	-------
	cpfactor_list    factor matrix list consist of coefficients, len(cpfactor_list) = CP rank 
	sample_point    a vector
	num_basis    the number of basis, each basis is a Legendre polynomial

	returns:
	--------
	function values at sample_point
	"""
	cp_rank = len(cpfactor_list)
	f_value = 0
	Phi = np.zeros((len(sample_point), num_basis))
	for j in range(len(sample_point)):
		Phi[j,:] = construct_phi(num_basis, sample_point[j])
	for i in range(cp_rank):
		f_value += np.prod(np.sum(cpfactor_list[i]*Phi, axis=1))

	return f_value
