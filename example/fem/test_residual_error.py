#!/usr/bin/env python3
# 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_3d import CosCosCosData
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh 
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator 
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TetrahedronMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--nz',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
nz = args.nz
maxit = args.maxit


pde = CosCosCosData()
domain = pde.domain()

mesh = TetrahedronMesh.from_box(domain, nx=nx, ny=ny, nz=nz)

errorType = ['$|| u - u_h||_{\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$',
        'residual error']
errorMatrix = np.zeros((3, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator(q=p+2))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=p+2))
    F = lform.assembly()
    uI = space.interpolate(pde.solution)

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)
    err0 = np.max(np.abs(A@uI-F))
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = mesh.error(pde.solution, uh)
    errorMatrix[1, i] = mesh.error(pde.gradient, uh.grad_value)
    errorMatrix[2, i] = err0

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
print(np.log2(errorMatrix[:, 0:-1]/errorMatrix[:, 1:]))
