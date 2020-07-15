import numpy as np
from .TriangleMesh import TriangleMesh, TriangleMeshWithInfinityNode
from .TetrahedronMesh import TetrahedronMesh
from ..geometry import DistDomain2d, DistDomain3d
from ..geometry import dcircle, drectangle
from ..geometry import ddiff
from ..geometry import huniform
from .distmesh import DistMesh2d

class MeshFactory():
    def __init__(self):
        pass

    def one_triangle_mesh(self, meshtype='iso'):
        if meshtype == 'equ':
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2]], dtype=np.float64_)
        elif meshtype == 'iso':
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]], dtype=np.float64)
        cell = np.array([[0, 1, 2]], dtype=np.int_)
        return TriangleMesh(node, cell)

    def one_quad_mesh(self, meshtype='square'):
        if meshtype in {'square', 'zhengfangxing'}:
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0]], dtype=np.float64)
        elif mesthtype in {'rectangle', 'rec', 'juxing'}:
            node = np.array([
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0]], dtype=np.float64)
        elif meshtype in {'rhombus', 'lingxing'}:
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.5, np.sqrt(3)/2],
                [0.5, np.sqrt(3)/2]], dtype=np.float64)
        cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
        return QuadrangleMesh(node, cell)

    def one_tetrahedron_mesh(self, ttype='equ'):
        if ttype == 'equ':
            node = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3)/2, 0.0],
                [0.5, np.sqrt(3)/6, np.sqrt(2/3)]], dtype=np.float64)
        elif ttype == 'iso':
            node = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=np.float64)
        cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
        return TetrahedronMesh(node, cell)


    def boxmesh2d(box, nx=10, ny=10, meshtype='tri'):
        """

        Notes
        -----
        生成二维矩形区域上的网格，包括结构的三角形、四边形和三角形对偶的多边形网
        格
        """
        N = (nx+1)*(ny+1)
        NC = nx*ny
        node = np.zeros((N,2))
        X, Y = np.mgrid[box[0]:box[1]:complex(0,nx+1), box[2]:box[3]:complex(0,ny+1)]
        node[:,0] = X.flatten()
        node[:,1] = Y.flatten()

        idx = np.arange(N).reshape(nx+1, ny+1)
        if meshtype=='tri':
            cell = np.zeros((2*NC, 3), dtype=np.int)
            cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
            cell[:NC, 1] = idx[1:,1:].flatten(order='F')
            cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
            cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
            cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
            cell[NC:, 2] = idx[1:, 1:].flatten(order='F')
            return TriangleMesh(node, cell)
        elif meshtype == 'quad':
            cell = np.zeros((NC,4), dtype=np.int)
            cell[:,0] = idx[0:-1, 0:-1].flatten()
            cell[:,1] = idx[1:, 0:-1].flatten()
            cell[:,2] = idx[1:, 1:].flatten()
            cell[:,3] = idx[0:-1, 1:].flatten()
            return QuadrangleMesh(node, cell)
        elif meshtype == 'polygon':
            cell = np.zeros((NC,4), dtype=np.int)
            cell[:,0] = idx[0:-1, 0:-1].flatten()
            cell[:,1] = idx[1:, 0:-1].flatten()
            cell[:,2] = idx[1:, 1:].flatten()
            cell[:,3] = idx[0:-1, 1:].flatten()
            return PolygonMesh(node, cell)

    def boxmesh3d(self, box, nx=10, ny=10, nz=10, meshtype='hex'):
        """
        Notes
        -----
        生成长方体区域上的六面体或四面体网格。
        """
        N = (nx+1)*(ny+1)*(nz+1)
        NC = nx*ny*nz
        node = np.zeros((N, 3), dtype=np.float)
        X, Y, Z = np.mgrid[
                box[0]:box[1]:complex(0, nx+1), 
                box[2]:box[3]:complex(0, ny+1),
                box[4]:box[5]:complex(0, nz+1)
                ]
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        idx = np.arange(N).reshape(nx+1, ny+1, nz+1)
        c = idx[:-1, :-1, :-1]

        cell = np.zeros((NC, 8), dtype=np.int)
        nyz = (ny + 1)*(nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + nyz
        cell[:, 2] = cell[:, 1] + nz + 1
        cell[:, 3] = cell[:, 0] + nz + 1
        cell[:, 4] = cell[:, 0] + 1
        cell[:, 5] = cell[:, 4] + nyz
        cell[:, 6] = cell[:, 5] + nz + 1
        cell[:, 7] = cell[:, 4] + nz + 1
        if meshtype == 'hex':
            return HexahedronMesh(node, cell)
        elif meshtype == 'tet':
            localCell = np.array([
                [0, 1, 2, 6],
                [0, 5, 1, 6],
                [0, 4, 5, 6],
                [0, 7, 4, 6],
                [0, 3, 7, 6],
                [0, 2, 3, 6]], dtype=np.int)
            cell = cell[:, localCell].reshape(-1, 4)
            return TetrahedronMesh(node, cell)

    def triangle(box, h, meshtype='tri'):
        """
        Notes
        -----
        生成非矩形区域上的非结构网格网格
        """
        from meshpy.triangle import MeshInfo, build
        mesh_info = MeshInfo()
        mesh_info.set_points([(box[0], box[2]), (box[1], box[2]), (box[1], box[3]), (box[0], box[3])])
        mesh_info.set_facets([[0, 1], [1, 2], [2, 3], [3, 0]])
        mesh = build(mesh_info, max_volume=h**2)
        node = np.array(mesh.points, dtype=np.float)
        cell = np.array(mesh.elements, dtype=np.int)
        if meshtype == 'tri':
            return TriangleMesh(node, cell)
        elif meshtype == 'polygon':
            mesh = TriangleMeshWithInfinityNode(TriangleMesh(node, cell))
            pnode, pcell, pcellLocation = mesh.to_polygonmesh()
            return PolygonMesh(pnode, pcell, pcellLocation)

    def special_box_mesh2d(self, box, n=10, meshtype='fishbone'):
        qmesh = self.boxmesh2d(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NN = qmesh.number_of_nodes()
        NE = qmesh.number_of_edges()
        NC = qmesh.number_of_cells()
        if meshtype == 'fishbone':
            isLeftCell = np.zeros((n, n), dtype=np.bool)
            isLeftCell[0::2, :] = True
            isLeftCell = isLeftCell.reshape(-1)
            lcell = cell[isLeftCell]
            rcell = cell[~isLeftCell]
            newCell = np.r_['0', 
                    lcell[:, [1, 2, 0]], 
                    lcell[:, [3, 0, 2]],
                    rcell[:, [0, 1, 3]],
                    rcell[:, [2, 3, 1]]]
            return TriangleMesh(node, newCell)
        elif meshtype == 'cross': 
            bc = qmesh.barycenter('cell') 
            newNode = np.r_['0', node, bc]

            newCell = np.zeros((4*NC, 3), dtype=np.int) 
            newCell[0:NC, 0] = range(NN, NN+NC)
            newCell[0:NC, 1:3] = cell[:, 0:2]
            
            newCell[NC:2*NC, 0] = range(NN, NN+NC)
            newCell[NC:2*NC, 1:3] = cell[:, 1:3]

            newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
            newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

            newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
            newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
            return TriangleMesh(newNode, newCell)
        elif meshtype == 'rice':
            isLeftCell = np.zeros((n, n), dtype=np.bool)
            isLeftCell[0, 0::2] = True
            isLeftCell[1, 1::2] = True
            if n > 2:
                isLeftCell[2::2, :] = isLeftCell[0, :]
            if n > 3:
                isLeftCell[3::2, :] = isLeftCell[1, :]
            isLeftCell = isLeftCell.reshape(-1)

            lcell = cell[isLeftCell]
            rcell = cell[~isLeftCell]
            newCell = np.r_['0', 
                    lcell[:, [1, 2, 0]], 
                    lcell[:, [3, 0, 2]],
                    rcell[:, [0, 1, 3]],
                    rcell[:, [2, 3, 1]]]
            return TriangleMesh(node, newCell)
        elif meshtype == 'nonuniform':
            nx = 4*n
            ny = 4*n
            n1 = 2*n+1

            N = n1**2
            NC = 4*n*n
            node = np.zeros((N,2))
            
            x = np.zeros(n1, dtype=np.float64)
            x[0::2] = range(0,nx+1,4)
            x[1::2] = range(3, nx+1, 4)

            y = np.zeros(n1, dtype=np.float64)
            y[0::2] = range(0,nx+1,4)
            y[1::2] = range(1, nx+1,4)

            node[:,0] = x.repeat(n1)/nx
            node[:,1] = np.tile(y, n1)/ny


            idx = np.arange(N).reshape(n1, n1)
            
            cell = np.zeros((2*NC, 3), dtype=np.int_)
            cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
            cell[:NC, 1] = idx[1:,1:].flatten(order='F')
            cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
            cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
            cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
            cell[NC:, 2] = idx[1:, 1:].flatten(order='F')
            return TriangleMesh(node, cell)
        
    def lshape_mesh(self, n=4):
        point = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float)

        cell = np.array([
            (1, 3, 0),
            (2, 0, 3),
            (3, 6, 2),
            (5, 2, 6),
            (4, 7, 3),
            (6, 3, 7)], dtype=np.int)
        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n)
        return mesh

    def unitcirclemesh(self, h0, meshtype='tri'):
        """

        Reference
        ---------

        Notes
        -----
        利用 distmesh 算法生成单位圆上的非结构三角形或多边形网格
        """
        fd = lambda p: dcircle(p, (0,0), 1)
        fh = huniform
        bbox = [-1.2, 1.2, -1.2, 1.2]
        pfix = None 
        domain = DistDomain2d(fd, fh, bbox, pfix)
        distmesh2d = DistMesh2d(domain, h0)
        distmesh2d.run()
        if meshtype == 'tri':
            return distmesh2d.mesh
        elif meshtype == 'polygon':
            mesh = TriangleMeshWithInfinityNode(distmesh2d.mesh)
            pnode, pcell, pcellLocation = mesh.to_polygonmesh()
            return PolygonMesh(pnode, pcell, pcellLocation) 
       
# 下面的程序还需要标准化
    def uncross_mesh(self, box, n=10, r="1"):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NN = qmesh.number_of_nodes()
        NC = qmesh.number_of_cells()   
        bc = qmesh.barycenter('cell') 

        if r=="1":
            bc1 = np.sqrt(np.sum((bc-node[cell[:,0], :])**2, axis=1))[0]
            newNode = np.r_['0', node, bc-bc1*0.3]
        elif r=="2":
            ll = node[cell[:, 0]] - node[cell[:, 2]]
            bc = qmesh.barycenter('cell') + ll/4
            newNode = np.r_['0',node, bc]

        newCell = np.zeros((4*NC, 3), dtype=np.int) 
        newCell[0:NC, 0] = range(NN, NN+NC)
        newCell[0:NC, 1:3] = cell[:, 0:2]
            
        newCell[NC:2*NC, 0] = range(NN, NN+NC)
        newCell[NC:2*NC, 1:3] = cell[:, 1:3]

        newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
        newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

        newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
        newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
        return TriangleMesh(newNode, newCell)




