#!/usr/bin/python3
# import ipdb
from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('INFO')
from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.pde.poisson_3d import CosCosCosData
from fealpy.mesh import TriangleMesh,TetrahedronMesh
from fealpy.fem import PoissonLFEMSolver
import time
from fealpy.sparse import CSRTensor
bm.set_backend('pytorch')
bm.set_default_device('cuda')


# 初始化计时器和日志
tmr = timer()
next(tmr)

p = 1
# 定义测试配置 (n, m)
configurations = [
    (2, 10),  # 配置 1
    # (8, 8),   # 配置 2
    # (32, 6)   # 配置 3
]

# 定义 PDE 问题
# pde = CosCosCosData()
pde = CosCosData()
domain = pde.domain()

# 遍历每种配置
for n, m in configurations:
    # 创建初始网格
    mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
    # mesh = TetrahedronMesh.from_box(box=domain,nx=n,ny=n,nz=n)
    
    # 均匀加密网格
    IM = mesh.uniform_refine(n=m, returnim=True)
    
    # 初始化求解器
    p = 1  # 多项式阶数
    s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)
    
    # 运行 gamg_solve
    tmr.send(f"Running gamg_solve with (n, m) = ({n}, {m})")
    # s0.gamg_solve(IM)
    import cProfile
    cProfile.run('s0.gamg_solve(IM)')
    
    
    # 记录当前配置完成
    tmr.send(f"Completed gamg_solve for (n, m) = ({n}, {m})")

# 结束计时器
tmr.send(None)

