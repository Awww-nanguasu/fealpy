import ipdb
import argparse
import sympy as sp
import matplotlib.pyplot as plt
from scipy.sparse import coo_array, csr_array, bmat
from mpl_toolkits.mplot3d import Axes3D

from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm

from fealpy.pde.surface_poisson_model import SurfaceLevelSetPDEData
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.tools.show import showmultirate, show_error_table

# solver
from fealpy.solver import cg, spsolve

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        曲面上的任意次等参有限元方法
        """)

parser.add_argument('--sdegree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--mdegree',
        default=1, type=int,
        help='网格的阶数, 默认为 1 次.')

parser.add_argument('--mtype',
        default='ltri', type=str,
        help='网格类型， 默认三角形网格.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy.")

parser.add_argument('--maxit',
        default=4, type=int,
        help="默认网格加密求解次数，默认加密求解4次.")

args = parser.parse_args()
bm.set_backend(args.backend)

sdegree = args.sdegree
mdegree = args.mdegree
mtype = args.mtype
maxit = args.maxit

x, y, z = sp.symbols('x, y, z', real=True)
F = x**2 + y**2 + z**2
u = x * y
pde = SurfaceLevelSetPDEData(F, u)

p = mdegree
surface = SphereSurface()
tmesh = TriangleMesh.from_unit_sphere_surface()


errorType = ['$|| u - u_h||_{\Omega,0}$']
errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
NDof = bm.zeros(maxit, dtype=bm.int32)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    
    mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh, p=mdegree, surface=surface)
    #fname = f"sphere_test.vtu"
    #mesh.to_vtk(fname=fname)
    
    space = ParametricLagrangeFESpace(mesh, p=sdegree)
    NDof[i] = space.number_of_global_dofs()

    uI = space.interpolate(pde.solution)

    #ipdb.set_trace()
    bfrom = BilinearForm(space)
    bfrom.add_integrator(ScalarDiffusionIntegrator(method='isopara'))
    lfrom = LinearForm(space)
    lfrom.add_integrator(ScalarSourceIntegrator(pde.source))

    A = bfrom.assembly(format='coo')
    F = lfrom.assembly()
    C = space.integral_basis()

    def coo(A):
        data = A._values
        indices = A._indices
        return coo_array((data, indices), shape=A.shape)
    A = bmat([[coo(A), C.reshape(-1,1)], [C, None]], format='coo')
    A = COOTensor(bm.stack([A.row, A.col], axis=0), A.data, spshape=A.shape)

    F = bm.concatenate((F, bm.array([0])))
    
    uh = space.function()
    x = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14).reshape(-1)
    uh[:] = -x[:-1]

    #uh[:] = -spsolve(A, F)[:-1]
    
    #tmr.send(f'第{i}次求解器时间')

    errorMatrix[0, i] = mesh.error(pde.solution, uh.value, q=p+3)

    if i < maxit-1:
        tmesh.uniform_refine()

print("最终误差:", errorMatrix)
print("order:", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 2, NDof, errorMatrix,  errorType, propsize=20)
plt.show()