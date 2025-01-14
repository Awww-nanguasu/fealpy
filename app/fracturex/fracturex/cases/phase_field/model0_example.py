import numpy as np
import torch
import argparse

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.old.geometry.domain_2d import SquareWithCircleHoleDomain

from app.fracturex.fracturex.phasefield.main_solve import MainSolve

from fealpy.utils import timer

import time
import matplotlib.pyplot as plt

class square_with_circular_notch():
    def __init__(self):
        """
        @brief 初始化模型参数
        """
        E = 200
        nu = 0.2
        Gc = 1.0
        l0 = 0.02
        self.params = {'E': E, 'nu': nu, 'Gc': Gc, 'l0': l0}


    def is_force(self):
        """
        @brief 位移增量条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return bm.concatenate((bm.linspace(0, 70e-3, 6, dtype=bm.float64), bm.linspace(70e-3,
            125e-3, 26, dtype=bm.float64)[1:]))

    def is_force_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = bm.abs(p[..., 1] - 1) < 1e-12 
        return isDNode

    def is_dirchlet_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return bm.abs((p[..., 0]-0.5)**2 + bm.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        脆性断裂任意次自适应有限元
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--maxit',
        default=30, type=int,
        help='最大迭代次数, 默认为 30 次.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='有限元计算后端, 默认为 numpy.')

parser.add_argument('--model_type',
        default='HybridModel', type=str,
        help='有限元方法, 默认为 HybridModel.')

parser.add_argument('--mesh_type',
        default='tri', type=str,
        help='网格类型, 默认为 tri.')

parser.add_argument('--enable_adaptive',
        default=False, type=bool,
        help='是否启用自适应加密, 默认为 False.')

parser.add_argument('--marking_strategy',
        default='recovery', type=str,
        help='标记策略, 默认为重构型后验误差估计.')

parser.add_argument('--refine_method',
        default='bisect', type=str,
        help='网格加密方法, 默认为 bisect.')

parser.add_argument('--h',
        default=0.01, type=float,
        help='初始网格最小尺寸, 默认为 0.01.')

parser.add_argument('--vtkname',
        default='test', type=str,
        help='vtk 文件名, 默认为 test.')

parser.add_argument('--save_vtkfile',
        default=True, type=bool,
        help='是否保存 vtk 文件, 默认为 False.')


parser.add_argument('--atype', 
        default=None, type=str,
        help='矩阵组装的方法, 默认为 常规组装.')

parser.add_argument('--gpu', 
        default=False, type=bool,
        help='是否使用 GPU, 默认为 False.')

args = parser.parse_args()
p= args.degree
maxit = args.maxit
backend = args.backend
model_type = args.model_type
enable_adaptive = args.enable_adaptive
marking_strategy = args.marking_strategy
refine_method = args.refine_method
h = args.h
save_vtkfile = args.save_vtkfile
vtkname = args.vtkname +'_' + args.mesh_type + '_'
atype = args.atype
gpu = args.gpu

tmr = timer()
next(tmr)
start = time.time()
bm.set_backend(backend)

if gpu:
    bm.set_default_device('cuda')

model = square_with_circular_notch()

domain = SquareWithCircleHoleDomain(hmin=h) 
mesh = TriangleMesh.from_domain_distmesh(domain, maxit=100)


ms = MainSolve(mesh=mesh, material_params=model.params)
tmr.send('init')

# 拉伸模型边界条件
ms.add_boundary_condition('force', 'Dirichlet', model.is_force_boundary, model.is_force(), 'y')

# 固定位移边界条件
ms.add_boundary_condition('displacement', 'Dirichlet', model.is_dirchlet_boundary, 0)
ms.add_boundary_condition('phase', 'Dirichlet', model.is_dirchlet_boundary, 0)

if atype == 'auto':
    ms.auto_assembly_matrix()
elif atype == 'fast':
    ms.fast_assembly_matrix()
    

#ms.set_scipy_solver()
ms.output_timer()

ms.save_vtkfile(fname=vtkname)
ms.solve(p=p, maxit=maxit)

tmr.send('stop')
tmr.send(None)
end = time.time()

force = ms.get_residual_force()
disp = model.is_force()

ftname = 'force_'+args.mesh_type + '_p' + str(p) + '_' + 'model0_disp.pt'

torch.save(force, ftname)
#np.savetxt('force'+tname, bm.to_numpy(force))
tname = 'params_'+args.mesh_type + '_p' + str(p) + '_' + 'model0_disp.txt'
with open(tname, 'w') as file:
    file.write(f'time: {end-start},\n degree:{p},\n, backend:{backend},\n, model_type:{model_type},\n, enable_adaptive:{enable_adaptive},\n, marking_strategy:{marking_strategy},\n, refine_method:{refine_method},\n, hmin:{h},\n, maxit:{maxit},\n, vtkname:{vtkname}\n')
fig, axs = plt.subplots()
plt.plot(disp, force, label='Force')
plt.xlabel('Displacement Increment')
plt.ylabel('Residual Force')
plt.title('Changes in Residual Force')
plt.grid(True)
plt.legend()
pname = args.mesh_type + '_p' + str(p) + '_' + 'model0_force.png'
plt.savefig(pname, dpi=300)

print(f"Time: {end - start}")
