import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats
from functools import partial

def expectile_loss(diff, tau):
    '''
    diff: x - y, where x is prediction and y is the label
    tau: the percentile to model for the expectile
    '''
    if isinstance(diff, np.ndarray):
        return np.abs(tau - np.where(diff < 0, 1, 0)) * np.square(diff)
    else:
        return torch.clamp(torch.abs(tau - torch.where(diff < 0, 1, 0)) * torch.square(diff), min=-1, max=5)

def gumbel_max_loss(diff, beta):
    '''
    diff: x - y, where x is prediction and y is the label
    beta: the scale parameter for the gumbel
    '''
    if isinstance(diff, np.ndarray):
        return diff/beta + np.exp(-1*diff/beta) - 1
    else:
        return diff/beta + torch.exp(-1*diff/beta) - 1

def gumbel_min_loss(diff, beta):
    '''
    diff: x - y, where x is prediction and y is the label
    beta: the scale parameter for the gumbel
    '''
    if isinstance(diff, np.ndarray):
        return -diff/beta + np.exp(diff/beta) - 1
    else:
        return -diff/beta + torch.exp(diff/beta) - 1

def solver_1d(data, loss_fn, lr=0.005, steps=4000):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    statistic = torch.zeros(1, requires_grad=True)
    optim = torch.optim.SGD([statistic], lr=lr)
    for _ in range(steps):
        optim.zero_grad()
        loss = torch.mean(loss_fn(data - statistic))
        loss.backward()
        optim.step()
    return statistic.detach().cpu().numpy()

def solver_gumbel(data, loss_fn, lr=0.005, steps=4000):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    mu = torch.zeros(1, requires_grad=True)
    beta = torch.ones(1, requires_grad=True)
    optim = torch.optim.SGD([mu, beta], lr=lr)
    for _ in range(steps):
        optim.zero_grad()
        loss = torch.mean(loss_fn(data - mu, beta))
        loss.backward()
        optim.step()
    return mu.detach().cpu().numpy(), beta.detach().cpu().numpy()

def solver_quadratic(x, y, loss_fn, lr=0.005, steps=4000):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    params = torch.randn(3, requires_grad=True)
    optim = torch.optim.SGD([params], lr=lr)
    x_squared = x*x
    for _ in range(steps):
        pred = params[0]*x_squared + params[1]*x + params[2]
        optim.zero_grad()
        loss = torch.mean(loss_fn(y - pred))
        loss.backward()
        optim.step()
    return params.detach().cpu().numpy()


def solver_gumbel_quadratic(x, y, loss_fn, lr=0.005, steps=4000):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    mu_params = torch.randn(3, requires_grad=True)
    beta = torch.ones(1, requires_grad=True)
    optim = torch.optim.SGD([mu_params, beta], lr=lr)
    x_squared = x*x
    for _ in range(steps):
        pred = mu_params[0]*x_squared + mu_params[1]*x + mu_params[2]
        optim.zero_grad()
        loss = torch.mean(loss_fn(y - pred, beta))
        loss.backward()
        optim.step()
    return mu_params.detach().cpu().numpy(), beta.detach().cpu().numpy()

def plot_quardatic(coeffs, domain=[-1, 1], pts=100, label=None):
    x = np.linspace(domain[0], domain[1], pts)
    y = coeffs[0]*x*x + coeffs[1]*x + coeffs[2]
    plt.plot(x, y, label=label)

def generate_quadratic_data():
    coeffs = [-1, 0, 1]
    x = np.linspace(-1, 1, 2000)
    y = coeffs[0]*x*x + coeffs[1]*x + coeffs[2]
    y += np.random.normal(loc=0.0, scale=0.2, size=y.shape[0])
    return x, y

'''
Expectile Plots
'''

def plot_expectile_loss():
    taus = [0.01, 0.1, 0.5, 0.9, 0.99]
    plt.clf()
    diff = np.linspace(-1, 1, 200)
    for tau in taus:
        loss = expectile_loss(diff, tau)
        plt.plot(diff, loss, label="tau="+str(tau))
    plt.title("Expectile Loss")
    plt.xlabel("Residual = x - y")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_expectile_normal():
    '''
    Find the expectiles via random sampling.
    '''
    plt.clf()
    fig, ax = plt.subplots()
    taus = [0.01, 0.1, 0.5, 0.9, 0.99]
    x = np.linspace(-3, 3, 500)
    y = stats.norm.pdf(x, loc=0.0, scale=1.0)
    plt.plot(x, y)
    data = np.random.normal(loc=0.0, scale=1.0, size=1000)
    for tau in taus:
        loss_fn = partial(expectile_loss, tau=tau)
        expectile = solver_1d(data, loss_fn, lr=0.02, steps=2000)
        plt.axvline(x=expectile, label="tau="+str(tau), color = next(ax._get_lines.prop_cycler)['color'])
    plt.title("Normal Expectiles")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.legend()
    plt.show()

def plot_expectile_quadratic():
    taus = [0.01, 0.1, 0.5, 0.9, 0.99]
    plt.clf()
    x, y = generate_quadratic_data()
    plt.scatter(x, y, s=0.5)
    for tau in taus:
        loss_fn = partial(expectile_loss, tau=tau)
        predicted_coeffs = solver_quadratic(x, y, loss_fn, lr=0.02, steps=50000)
        plot_quardatic(predicted_coeffs, label="tau="+str(tau))
    plt.title("Quadratic Expectile Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

'''
Gumbel Plots
'''

def plot_gumbel_loss(min=False):
    loss_fn = gumbel_min_loss if min else gumbel_max_loss
    betas = [0.5, 1, 2, 4]
    plt.clf()
    diff = np.linspace(-1, 1, 200)
    for beta in betas:
        loss = loss_fn(diff, beta)
        plt.plot(diff, loss, label="beta="+str(beta))
    plt.title("Gumbel " + ("min" if min else "max") + " Loss")
    plt.xlabel("Residual = x - y")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_gumbel_normal(min=False):
    loss_fn = gumbel_min_loss if min else gumbel_max_loss
    plt.clf()
    fig, ax = plt.subplots()
    betas = [0.25, 0.5, 1, 2]
    x = np.linspace(-3, 3, 500)
    y = stats.norm.pdf(x, loc=0.0, scale=1.0)
    plt.plot(x, y)
    data = np.random.normal(loc=0.0, scale=1.0, size=1000)
    for beta in betas:
        expectile = solver_1d(data, partial(loss_fn, beta=beta), lr=0.02, steps=2000)
        plt.axvline(x=expectile, label="beta="+str(beta), color = next(ax._get_lines.prop_cycler)['color'])
    plt.title("Gumbel " + ("min" if min else "max") + " Modes")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.legend()
    plt.show()


def plot_gumbel_quadratic(min=False):
    loss_fn = gumbel_min_loss if min else gumbel_max_loss
    betas = [0.15, 0.2, 0.25, 0.5, 1]
    plt.clf()
    x, y = generate_quadratic_data()
    plt.scatter(x, y, s=0.5)
    for beta in betas:
        predicted_coeffs = solver_quadratic(x, y, partial(loss_fn, beta=beta), lr=0.1, steps=50000)
        plot_quardatic(predicted_coeffs, label="beta="+str(beta))
    plt.title("Quadratic Gumbel " + ("min" if min else "max") + " Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-0.5, 1.5)
    plt.legend()
    plt.show()

def plot_gumbel_mle_normal():
    plt.clf()
    fig, ax = plt.subplots()
    x = np.linspace(-3, 3, 500)
    y = stats.norm.pdf(x, loc=0.0, scale=1.0)
    plt.plot(x, y)
    data = np.random.normal(loc=0.0, scale=1.0, size=1000)

    max_mode, max_beta = solver_gumbel(data, gumbel_max_loss, lr=0.02, steps=2000)
    plt.axvline(x=max_mode, label="Max Gumbel, learned_beta="+str(max_beta), color = next(ax._get_lines.prop_cycler)['color'])
    min_mode, min_beta = solver_gumbel(data, gumbel_min_loss, lr=0.02, steps=2000)
    plt.axvline(x=min_mode, label="Min Gumbel, learned_beta="+str(min_beta), color = next(ax._get_lines.prop_cycler)['color'])
    
    plt.title("Normal Gumbel MLE")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.legend()
    plt.show()

def plot_gumbel_mle_quadratic(min=False):
    plt.clf()
    x, y = generate_quadratic_data()
    plt.scatter(x, y, s=0.5)

    predicted_coeffs, beta = solver_gumbel_quadratic(x, y, gumbel_max_loss, lr=0.02, steps=10000)
    plot_quardatic(predicted_coeffs, label="Max beta="+str(beta))
    predicted_coeffs, beta = solver_gumbel_quadratic(x, y, gumbel_min_loss, lr=0.02, steps=10000)
    plot_quardatic(predicted_coeffs, label="Min beta="+str(beta))
    
    plt.title("Quadratic MLE Gumbel Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-0.5, 1.5)
    plt.legend()
    plt.show()

def plot_max_reg():
    plt.clf()
    coeffs = [-1, 0, 1]
    num_samples = 50
    x = np.linspace(-1, 1, 100)
    y = coeffs[0]*x*x + coeffs[1]*x + coeffs[2]
    full_xs = np.tile(x, (num_samples, 1)).T
    full_ys = np.expand_dims(y, -1) + np.random.normal(loc=0.0, scale=0.2, size=(y.shape[0], num_samples))
    max_ys = np.max(full_ys, axis=1)
    
    plt.scatter(full_xs.flatten(), full_ys.flatten(), s=0.5)
    plt.scatter(x, max_ys, s=0.75)

    predicted_coeffs, beta = solver_gumbel_quadratic(x, max_ys, gumbel_max_loss, lr=0.02, steps=10000)
    plot_quardatic(predicted_coeffs, label="Max beta="+str(beta))
    predicted_coeffs, beta = solver_gumbel_quadratic(x, max_ys, gumbel_min_loss, lr=0.02, steps=10000)
    plot_quardatic(predicted_coeffs, label="Min beta="+str(beta))

    predicted_coeffs = solver_quadratic(x, max_ys, partial(expectile_loss, tau=0.5), lr=0.02, steps=50000)
    plot_quardatic(predicted_coeffs, label="LS Quadratic Reg")

    plt.title("Quadratic MLE Gumbel Regression on Maximal Data")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.ylim(-0.5, 1.5)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # # Loss Plots
    plot_expectile_loss()
    plot_gumbel_loss(min=True)

    # # Normal Plots
    plot_expectile_normal()
    plot_gumbel_normal(min=True)
    plot_gumbel_mle_normal()

    # Quadratic Plots
    plot_expectile_quadratic()
    plot_gumbel_quadratic(min=True)
    plot_gumbel_mle_quadratic()
