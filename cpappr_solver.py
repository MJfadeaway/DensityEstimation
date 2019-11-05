from __future__ import division
import numpy as np
from cp_utils import *


# Author: Kejun Tang
# Last Revised: September 29, 2018

class Solver():
	"""Solver encapsulates all training details for continuous CP decomposition for function approximation"""

	def __init__(self, data, y, cpfactor, num_basis, **kwargs):
		"""
		data    training data matrix, numpy array. Each row is a sample drawn from an unknown distribution
		init_cpfactor    a list, initialization for continuous CP factor, store all the coefficients of Legendre polynomials
		num_basis    each CP factor (univariate function) has num_basis Legendre polynomial basis
		"""
		self.data = data
		self.y = y
		self.cpfactor = cpfactor
		self.num_basis = num_basis
		# unpack kwargs parameters
		self.batch_size = kwargs.pop('batch_size', 100)
		self.optim_configs = [None for i in range(len(cpfactor))]
		self.lr_decay = kwargs.pop('lr_decay', 1.0) # learning rate decay in each epoch
		self.num_epochs = kwargs.pop('num_epochs', 500)
		self.print_every = kwargs.pop('print_every', 10)
		if cpfactor[1].shape[1] != num_basis:
			raise ValueError('CP factor columns {}  does not match num_basis {}'.format(cpfacator[1].shape[1], num_basis))
		self._reset()

	def _reset(self):
		"""for training process"""
		self.loss_history = []
		self.epoch = 0


	def _step(self):
		"""step train function, called by train function"""
		cpfactor = self.cpfactor
		data = self.data
		y = self.y
		batch_mask = np.random.choice(data.shape[0], self.batch_size) # for batch gradient 
		batch_data = data[batch_mask]
		batch_y = y[batch_mask]
		grads = mse_gradient(cpfactor, batch_data, batch_y, self.num_basis)
		mse_loss = loss_funappr(cpfactor, batch_data, batch_y, self.num_basis)
		self.loss_history.append(mse_loss)
		# update 
		for p in range(len(cpfactor)):
			optim_config = self.optim_configs[p]
			next_x, next_config = nll_Adam(cpfactor[p], grads[p], optim_config)
			self.cpfactor[p] = next_x
			self.optim_configs[p] = next_config

	"""
	def compute_nllloss(self, train_data):
		# compute negative log likelihood function during training process
		cpfactor = self.cpfactor
		num_basis = self.num_basis
		nllloss = neg_loglikeli(cpfactor, train_data, num_basis)
		return nllloss
	"""

	def train(self):
		"""training process"""
		"""
		using Adam optimization algorithm to minimize loss function
		"""
		num_data = self.data.shape[0]
		iter_per_epoch = max(num_data//self.batch_size, 1) # iteration in every epoch
		num_iters = self.num_epochs * iter_per_epoch # total iterations

		for idx_iter in range(num_iters):
			self._step()
			if idx_iter % self.print_every == 0:
				print('iteration: {}, mse_loss: {:.4}'. format(idx_iter, self.loss_history[-1]))

			epoch_end = (idx_iter + 1) % iter_per_epoch == 0
			if epoch_end:
				self.epoch += 1
				for k in range(len(self.cpfactor)):
					self.optim_configs[k]['learning_rate'] *= self.lr_decay
