import numpy as np


class DiffusionIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """    
    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):
        """
        @note 没有参考单元的组装方式
        """
        coef = self.coef
        q = self.q
        mesh = space.mesh
        GD = mesh.geo_dimension()
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)
        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        if out is None:
            D = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            D = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi0 = space.grad_basis(bcs, index=index) # (NQ, NC, ldof, ...)
        phi1 = phi0


        if coef is None:
            D += np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs, index=index)
                        coef = coef(ps)
                    elif coef.coordtype == 'barycentric':
                        coef = coef(bcs, index=index)
                else:
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            if np.isscalar(coef):
                D += coef*np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
            elif isinstance(coef, np.ndarray): 
                if coef.shape == (NC, ): 
                    D += np.einsum('q, c, qcim, qcjm, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                elif coef.shape == (NQ, NC):
                    D += np.einsum('q, qc, qcim, qcjm, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                else:
                    n = len(coef.shape)
                    shape = (4-n)*(1, ) + coef.shape
                    D += np.einsum('q, qcmn, qcin, qcjm, c->cij', ws, coef.reshape(shape), phi0, phi1, cellmeasure, optimize=True)
            else:
                raise ValueError("coef不支持该类型")

        if out is None:
            return D


    def assembly_cell_matrix_fast(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space0.mesh 
        assert mesh.meshtype in ['tri', 'tet']


    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元矩阵组装方式
        """
