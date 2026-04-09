import numpy as np
import deepxde as dde
from deepxde.backend import pytorch as bkd
import torch


L = 200.0


def spr_true(x):
    return 1.0 + 0.3 * torch.sin(np.pi * x[:, 0:1] / L)


def pde(x, y):
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    return dy_dx - spr_true(x)


def ic(x):
    return y_true(x)[:, 0:1]


def bc_func(x):
    return torch.zeros_like(x[:, 0:1])


def y_true(x):
    x_t = x[:, 0:1]
    term1 = x_t
    term2 = -0.3 * (L / np.pi) * torch.cos(np.pi * x_t / L)
    term0 = 0.3 * (L / np.pi)
    return term1 + term2 + term0


geom = dde.geometry.Interval(0.0, L)


def main():
    bc = dde.icbc.DirichletBC(geom, bc_func, lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0))
    data = dde.data.PDE(geom, pde, [bc], num_domain=200, num_boundary=2, solution=y_true)
    net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(epochs=20000)
    X = np.linspace(0.0, L, 201)[:, None]
    y_pred = model.predict(X)
    np.savez("pinn_wet_solution.npz", x=X, y=y_pred)


if __name__ == "__main__":
    main()
