from __future__ import division
import numpy as np
import cpappr_solver as cps
from cp_utils import *
import matplotlib.pyplot as plt



# three simple test functions
def sphere(x):
    """sphere function"""
    """
    sphere function f(x) = np.sum(x**2)

    inputs:
    -------
    x    a matrix, each row is a sample point

    returns:
    --------
    sphere function value at x
    """
    return np.sum(x**2, axis=1)


def rastrigin(x):
    """Rastrigin function"""
    """
    Rastrigin function f(x) = A*len(x) + np.sum(x**2 - A * np.cos(2*np.pi*x))
    here, A = 10

    inputs:
    -------
    x    a matrix, each row is a sample point

    returns:
    --------
    Rastrigin function at x
    """
    return 10*x.shape[1] + np.sum(x**2 - 10*np.cos(2*np.cos(2*np.pi*x)), axis=1)


def rosenbrock(x):
    """Rosenbrock function"""
    """
    Rosenbrock function

    inputs:
    -------
    x    a matrix, each row is a sample point

    returns:
    -------
    Rosenbrock function at x
    """
    return np.sum(100.0 * (x.T[1:] - x.T[:-1]**2.0)**2.0 + (1 - x.T[:-1])**2.0, axis=0)


# test dimension
d = 2
a = -5.12
b = 5.12
test_type = 'sphere'
total_data_list = [100, 500, 1000, 1500, 2000, 2500]
train_error_list = []
test_error_list = np.zeros(len(total_data_list))

cpfactor = []
cp_rank = 10
num_basis = 5
for k in range(cp_rank):
    factor = np.random.rand(d, num_basis)
    cpfactor.append(factor)

parameter_dict = {'batch_size': 20, 'lr_decay': 1.0, 'num_epochs': 10, 'print_every': 10}

for k in range(len(total_data_list)):
    x = a + (b-a) * np.random.rand(total_data_list[k], d)
    if test_type == 'sphere':
        y = sphere(x)
    elif test_type == 'rastrigin':
        y = rastrigin(x)
    else: # rosenbrock
        y = rosenbrock(x)
    # split x, y to generate train data
    train_mask = np.random.choice(total_data_list[k], np.int(0.8*total_data_list[k]))
    test_mask = np.setdiff1d(range(total_data_list[k]), train_mask)
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]

    cps_pack = cps.Solver(x_train, y_train, cpfactor, num_basis, **parameter_dict)
    cps_pack.train()
    final_cpfactor = cps_pack.cpfactor
    train_loss = cps_pack.loss_history
    train_error_list.append(train_loss)

    ytest_cp = np.zeros(len(test_mask))
    for idx_sample in range(len(test_mask)):
        ytest_cp[idx_sample] = eval_cptensor(final_cpfactor, x_test[idx_sample], num_basis)

    test_error_list[k] = .5*np.linalg.norm(ytest_cp-y_test)**2/(len(test_mask))

plt.plot(total_data_list, test_error_list)
plt.show()
