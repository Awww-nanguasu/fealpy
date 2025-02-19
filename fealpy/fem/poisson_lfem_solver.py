
from ..backend import backend_manager as bm

from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..fem import DirichletBC

from fealpy.sparse import csr_matrix



class PoissonLFEMSolver:
    """
    """
    def __init__(self, pde, mesh, p, timer=None, logger=None):
        """
        """
        # 计时与日志
        self.timer = timer
        self.logger = logger

        self.p = p
        self.pde = pde
        self.mesh = mesh
        self.space = LagrangeFESpace(mesh, p=p)
        self.uh = self.space.function() # 建立一个有限元函数
        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))
        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source))
        A = bform.assembly()
        b = lform.assembly()
        if self.timer is not None:
            self.timer.send(f"组装 Poisson 方程离散系统")

        gdof = self.space.number_of_global_dofs()
        self.A, self.b = DirichletBC(self.space, gd=pde.solution).apply(A, b)
        if self.timer is not None:
            self.timer.send(f"处理 Poisson 方程 D 边界条件")


    def cg_solve(self):
        """
        """
        from ..solver import cg 
        # self.uh[:] = cg(self.A, self.b, maxit=5000, atol=1e-14, rtol=1e-14)
        self.uh[:],info = cg(self.A, self.b, maxit=5000, atol=1e-14, rtol=1e-14)
        
        if self.timer is not None:
            self.timer.send(f"CG 方法求解 Poisson 方程线性系统")
        # err = self.L2_error()
        # res = info['residual']
        # res_0 = bm.linalg.norm(self.b)
        # stop_res = res/res_0
        # self.logger.info(f"CG solver with {info['niter']} iterations"
        #                  f" and relative residual {stop_res:.4e},absolute error {err:.4e}")
        return self.uh

    # def gs_solve(self):
    #     from ..solver import gs

    #     self.uh[:] = gs(self.A, self.b, maxit=200, rtol=1e-8)
    #     err = self.L2_error()
    #     if self.timer is not None:
    #         self.timer.send(f"Gausss Seidel 方法求解 Poisson 方程线性系统")
    
    #     return self.uh
    
    # def jacobi_solve(self):
    #     from ..solver import jacobi

    #     self.uh[:] = jacobi(self.A, self.b, maxit=200, rtol=1e-8)
    #     err = self.L2_error()
    #     if self.timer is not None:
    #         self.timer.send(f"Jacobi 方法求解 Poisson 方程线性系统")

    #     return self.uh

    def gamg_solve(self, P=None, cdegree=[1]):
        """
        遍历 isolver ('MG', 'CG') 和 ptype ('V', 'F', 'W') 的所有组合，
        分别执行求解并记录结果
        """
        from ..solver import GAMGSolver
        isolver_options = ['MG']
        # ptype_options = ['V', 'F', 'W']
        ptype_options = ['W']
        for isolver in isolver_options:
            for ptype in ptype_options:
                # 初始化求解器组合
                # 记录当前组合的计时信息
                solver = GAMGSolver(isolver=isolver, ptype=ptype,device='cuda')
                
                # 原有逻辑保持不变
                if self.p < 2:
                    self.space = None
                solver.setup(self.A, P=P, space=self.space, cdegree=cdegree)
                if self.timer is not None:
                    self.timer.send(f"{isolver}-{ptype} 方法setup")

                
                self.uh[:], info = solver.solve(self.b)
                
                # 记录当前组合的计时信息
                if self.timer is not None:
                    self.timer.send(f"{isolver}-{ptype} 方法求解 Poisson 方程离散系统")
                
                # 计算误差和残差
                err = self.L2_error()
                res = info['residual']
                res_0 = bm.linalg.norm(self.b)
                stop_res = res / res_0
                
                # 在日志中标记当前组合
                self.logger.info(
                    f"{isolver}-{ptype}: {info['niter']} iterations, "
                    f"rel_res={stop_res:.4e}, abs_err={err:.4e}"
                )
                
    
        return self.uh[:]  # 返回最后一次解（或根据需求调整）
    # def gamg_solve(self, P=None, cdegree=[1]):
    #     """
    #     """
    #     from ..solver import GAMGSolver
    #     # solver = GAMGSolver(isolver='MG') 
    #     solver = GAMGSolver(isolver='MG') 
    #     if self.p < 2:
    #         self.space = None
    #     solver.setup(self.A, P=P, space=self.space, cdegree=cdegree)
    #     self.uh[:],info = solver.solve(self.b)
    #     if self.timer is not None:
    #         self.timer.send(f"MG 方法求解 Poisson 方程离散系统")
    #     err = self.L2_error()
    #     res = info['residual']
    #     res_0 = bm.linalg.norm(self.b)
    #     stop_res = res/res_0
    #     self.logger.info(f"MG solver with {info['niter']} iterations"
    #                      f" and relative residual {stop_res:.4e},absolute error {err:.4e}")

    #     return self.uh[:]

    def show_mesh_and_solution(self):
        """
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        mesh = self.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        fig = plt.figure()
        axes = fig.add_subplot(121)
        mesh.add_plot(axes)
        axes = fig.add_subplot(122, projection='3d')
        axes.plot_trisurf(node[:, 0], node[:, 1], self.uh, triangles=cell, cmap='rainbow')
        plt.show()


    def L2_error(self):
        """
        """
        return self.mesh.error(self.pde.solution, self.uh)



        


