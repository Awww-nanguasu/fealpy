import numpy as np

# 定义多个典型的 TriangleMesh 对象
init_data = [
    {
        "node": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [2, 0], [1, 2]], dtype=np.int32), 
        "cell": np.array([[0, 1, 2]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 2, 2], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.int32),
        "NN": 3,
        "NE": 3,
        "NF": 3,
        "NC": 1
    },
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [2, 0], [3, 0], [1, 2], [2, 3]], dtype=np.int32),
        "cell": np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 2, 2], [0, 0, 2, 2],[1, 1, 1, 1]], dtype=np.int32),
        "NN": 4,
        "NE": 5, 
        "NF": 5,
        "NC": 2
    }
]

from_one_triangle_data = [
        {
            "node": np.array([[0.       , 0.       ], [1.       , 0.       ],[0.5      , 0.8660254]], dtype=np.float64),
            "edge": np.array([[0, 1], [2, 0], [1, 2]], dtype=np.int32),
            "cell": np.array([[0, 1, 2]], dtype=np.int32),
            "face2cell": np.array([[0, 0, 2, 2],[0, 0, 1, 1],[0, 0, 0, 0]], dtype=np.int32),
            "NN": 3,
            "NE": 3,
            "NF": 3,
            "NC": 1
            }
]

from_box_data= [
            {
                "node": np.array([[0 , 0], [0, 0.5], [0, 1], [0.5, 0], 
                    [0.5, 0.5], [0.5, 1], [1, 0], [1, 0.5], [1, 1]], dtype=np.float64),
                "edge": np.array([[1, 0], [0, 3], [4, 0], [2, 1], [1, 4], [5, 1], 
                    [5, 2], [3, 4], [3, 6], [7, 3], [4, 5], [4, 7], [8, 4], [8, 5], 
                    [6, 7], [7, 8]], dtype=np.int32), 
                "cell": np.array([[3, 4, 0], [6, 7, 3], [4, 5, 1], [7, 8, 4], 
                    [1, 0, 4], [4, 3, 7], [2, 1, 5], [5, 4, 8]], dtype=np.int32),
                "face2cell": np.array([[4, 4, 2, 2], [0, 0, 1, 1], [0, 4, 0, 0], [6, 6, 2, 2],
                    [2, 4, 1, 1], [2, 6, 0, 0], [6, 6, 1, 1], [0, 5, 2, 2], [1, 1, 1, 1], 
                    [1, 5, 0, 0], [2, 7, 2, 2], [3, 5, 1, 1], [3, 7, 0, 0], [7, 7, 1, 1], 
                    [1, 1, 2, 2], [3, 3, 2, 2]], dtype=np.int32),
                "NN": 9,
                "NE": 16,
                "NF": 16,
                "NC": 8
            }
]

entity_measure_data = [
        {
            "node_measure": np.array([0.0], dtype=np.float64),
            "edge_measure": np.array([1.0, 1.0, np.sqrt(2)], dtype=np.float64),
            "cell_measure": np.array([0.5], dtype=np.float64)
            }
]

grad_lambda_data = [
        {
            "val": np.array([[[ 1., -1.],
            [ 0.,  1.],
            [-1.,  0.]],

           [[ 1., -1.],
            [ 0.,  1.],
            [-1.,  0.]],

           [[ 1., -1.],
            [ 0.,  1.],
            [-1.,  0.]],

           [[ 1., -1.],
            [ 0.,  1.],
            [-1.,  0.]],

           [[-1.,  1.],
            [ 0., -1.],
            [ 1.,  0.]],

           [[-1.,  1.],
            [ 0., -1.],
            [ 1.,  0.]],

           [[-1.,  1.],
            [ 0., -1.],
            [ 1.,  0.]],

           [[-1.,  1.],
            [ 0., -1.],
            [ 1.,  0.]]], dtype=np.float64)
       }
]

grad_shape_function_data = [
        {
            "gphi": np.array([[[[ 4.53478058, -4.53478058],
             [ 0.73260971,  5.80217088],
             [-5.80217088, -0.73260971],
             [ 0.        , -1.26739029],
             [-0.73260971,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[ 4.53478058, -4.53478058],
             [ 0.73260971,  5.80217088],
             [-5.80217088, -0.73260971],
             [ 0.        , -1.26739029],
             [-0.73260971,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[ 4.53478058, -4.53478058],
             [ 0.73260971,  5.80217088],
             [-5.80217088, -0.73260971],
             [ 0.        , -1.26739029],
             [-0.73260971,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[ 4.53478058, -4.53478058],
             [ 0.73260971,  5.80217088],
             [-5.80217088, -0.73260971],
             [ 0.        , -1.26739029],
             [-0.73260971,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[-4.53478058,  4.53478058],
             [-0.73260971, -5.80217088],
             [ 5.80217088,  0.73260971],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -0.73260971],
             [-1.26739029,  0.        ]],

            [[-4.53478058,  4.53478058],
             [-0.73260971, -5.80217088],
             [ 5.80217088,  0.73260971],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -0.73260971],
             [-1.26739029,  0.        ]],

            [[-4.53478058,  4.53478058],
             [-0.73260971, -5.80217088],
             [ 5.80217088,  0.73260971],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -0.73260971],
             [-1.26739029,  0.        ]],

            [[-4.53478058,  4.53478058],
             [-0.73260971, -5.80217088],
             [ 5.80217088,  0.73260971],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -0.73260971],
             [-1.26739029,  0.        ]]],


           [[[-1.26739029,  1.26739029],
             [ 6.53478058, -5.80217088],
             [ 0.        , -0.73260971],
             [ 0.        ,  4.53478058],
             [-6.53478058,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[-1.26739029,  1.26739029],
             [ 6.53478058, -5.80217088],
             [ 0.        , -0.73260971],
             [ 0.        ,  4.53478058],
             [-6.53478058,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[-1.26739029,  1.26739029],
             [ 6.53478058, -5.80217088],
             [ 0.        , -0.73260971],
             [ 0.        ,  4.53478058],
             [-6.53478058,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[-1.26739029,  1.26739029],
             [ 6.53478058, -5.80217088],
             [ 0.        , -0.73260971],
             [ 0.        ,  4.53478058],
             [-6.53478058,  0.73260971],
             [ 1.26739029,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-6.53478058,  5.80217088],
             [ 0.        ,  0.73260971],
             [ 0.        , -4.53478058],
             [ 6.53478058, -0.73260971],
             [-1.26739029,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-6.53478058,  5.80217088],
             [ 0.        ,  0.73260971],
             [ 0.        , -4.53478058],
             [ 6.53478058, -0.73260971],
             [-1.26739029,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-6.53478058,  5.80217088],
             [ 0.        ,  0.73260971],
             [ 0.        , -4.53478058],
             [ 6.53478058, -0.73260971],
             [-1.26739029,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-6.53478058,  5.80217088],
             [ 0.        ,  0.73260971],
             [ 0.        , -4.53478058],
             [ 6.53478058, -0.73260971],
             [-1.26739029,  0.        ]]],


           [[[-1.26739029,  1.26739029],
             [ 0.73260971,  0.        ],
             [ 5.80217088, -6.53478058],
             [ 0.        , -1.26739029],
             [-0.73260971,  6.53478058],
             [-4.53478058,  0.        ]],

            [[-1.26739029,  1.26739029],
             [ 0.73260971,  0.        ],
             [ 5.80217088, -6.53478058],
             [ 0.        , -1.26739029],
             [-0.73260971,  6.53478058],
             [-4.53478058,  0.        ]],

            [[-1.26739029,  1.26739029],
             [ 0.73260971,  0.        ],
             [ 5.80217088, -6.53478058],
             [ 0.        , -1.26739029],
             [-0.73260971,  6.53478058],
             [-4.53478058,  0.        ]],

            [[-1.26739029,  1.26739029],
             [ 0.73260971,  0.        ],
             [ 5.80217088, -6.53478058],
             [ 0.        , -1.26739029],
             [-0.73260971,  6.53478058],
             [-4.53478058,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-0.73260971,  0.        ],
             [-5.80217088,  6.53478058],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -6.53478058],
             [ 4.53478058,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-0.73260971,  0.        ],
             [-5.80217088,  6.53478058],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -6.53478058],
             [ 4.53478058,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-0.73260971,  0.        ],
             [-5.80217088,  6.53478058],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -6.53478058],
             [ 4.53478058,  0.        ]],

            [[ 1.26739029, -1.26739029],
             [-0.73260971,  0.        ],
             [-5.80217088,  6.53478058],
             [ 0.        ,  1.26739029],
             [ 0.73260971, -6.53478058],
             [ 4.53478058,  0.        ]]],


           [[[ 1.56758793, -1.56758793],
             [ 3.56758793,  0.        ],
             [-2.70276378, -0.86482415],
             [ 0.        ,  1.56758793],
             [-3.56758793,  0.86482415],
             [ 1.13517585,  0.        ]],

            [[ 1.56758793, -1.56758793],
             [ 3.56758793,  0.        ],
             [-2.70276378, -0.86482415],
             [ 0.        ,  1.56758793],
             [-3.56758793,  0.86482415],
             [ 1.13517585,  0.        ]],

            [[ 1.56758793, -1.56758793],
             [ 3.56758793,  0.        ],
             [-2.70276378, -0.86482415],
             [ 0.        ,  1.56758793],
             [-3.56758793,  0.86482415],
             [ 1.13517585,  0.        ]],

            [[ 1.56758793, -1.56758793],
             [ 3.56758793,  0.        ],
             [-2.70276378, -0.86482415],
             [ 0.        ,  1.56758793],
             [-3.56758793,  0.86482415],
             [ 1.13517585,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-3.56758793,  0.        ],
             [ 2.70276378,  0.86482415],
             [ 0.        , -1.56758793],
             [ 3.56758793, -0.86482415],
             [-1.13517585,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-3.56758793,  0.        ],
             [ 2.70276378,  0.86482415],
             [ 0.        , -1.56758793],
             [ 3.56758793, -0.86482415],
             [-1.13517585,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-3.56758793,  0.        ],
             [ 2.70276378,  0.86482415],
             [ 0.        , -1.56758793],
             [ 3.56758793, -0.86482415],
             [-1.13517585,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-3.56758793,  0.        ],
             [ 2.70276378,  0.86482415],
             [ 0.        , -1.56758793],
             [ 3.56758793, -0.86482415],
             [-1.13517585,  0.        ]]],


           [[[ 1.56758793, -1.56758793],
             [ 0.86482415,  2.70276378],
             [ 0.        , -3.56758793],
             [ 0.        , -1.13517585],
             [-0.86482415,  3.56758793],
             [-1.56758793,  0.        ]],

            [[ 1.56758793, -1.56758793],
             [ 0.86482415,  2.70276378],
             [ 0.        , -3.56758793],
             [ 0.        , -1.13517585],
             [-0.86482415,  3.56758793],
             [-1.56758793,  0.        ]],

            [[ 1.56758793, -1.56758793],
             [ 0.86482415,  2.70276378],
             [ 0.        , -3.56758793],
             [ 0.        , -1.13517585],
             [-0.86482415,  3.56758793],
             [-1.56758793,  0.        ]],

            [[ 1.56758793, -1.56758793],
             [ 0.86482415,  2.70276378],
             [ 0.        , -3.56758793],
             [ 0.        , -1.13517585],
             [-0.86482415,  3.56758793],
             [-1.56758793,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-0.86482415, -2.70276378],
             [ 0.        ,  3.56758793],
             [ 0.        ,  1.13517585],
             [ 0.86482415, -3.56758793],
             [ 1.56758793,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-0.86482415, -2.70276378],
             [ 0.        ,  3.56758793],
             [ 0.        ,  1.13517585],
             [ 0.86482415, -3.56758793],
             [ 1.56758793,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-0.86482415, -2.70276378],
             [ 0.        ,  3.56758793],
             [ 0.        ,  1.13517585],
             [ 0.86482415, -3.56758793],
             [ 1.56758793,  0.        ]],

            [[-1.56758793,  1.56758793],
             [-0.86482415, -2.70276378],
             [ 0.        ,  3.56758793],
             [ 0.        ,  1.13517585],
             [ 0.86482415, -3.56758793],
             [ 1.56758793,  0.        ]]],


           [[[-1.13517585,  1.13517585],
             [ 3.56758793, -2.70276378],
             [ 2.70276378, -3.56758793],
             [ 0.        ,  1.56758793],
             [-3.56758793,  3.56758793],
             [-1.56758793,  0.        ]],

            [[-1.13517585,  1.13517585],
             [ 3.56758793, -2.70276378],
             [ 2.70276378, -3.56758793],
             [ 0.        ,  1.56758793],
             [-3.56758793,  3.56758793],
             [-1.56758793,  0.        ]],

            [[-1.13517585,  1.13517585],
             [ 3.56758793, -2.70276378],
             [ 2.70276378, -3.56758793],
             [ 0.        ,  1.56758793],
             [-3.56758793,  3.56758793],
             [-1.56758793,  0.        ]],

            [[-1.13517585,  1.13517585],
             [ 3.56758793, -2.70276378],
             [ 2.70276378, -3.56758793],
             [ 0.        ,  1.56758793],
             [-3.56758793,  3.56758793],
             [-1.56758793,  0.        ]],

            [[ 1.13517585, -1.13517585],
             [-3.56758793,  2.70276378],
             [-2.70276378,  3.56758793],
             [ 0.        , -1.56758793],
             [ 3.56758793, -3.56758793],
             [ 1.56758793,  0.        ]],

            [[ 1.13517585, -1.13517585],
             [-3.56758793,  2.70276378],
             [-2.70276378,  3.56758793],
             [ 0.        , -1.56758793],
             [ 3.56758793, -3.56758793],
             [ 1.56758793,  0.        ]],

            [[ 1.13517585, -1.13517585],
             [-3.56758793,  2.70276378],
             [-2.70276378,  3.56758793],
             [ 0.        , -1.56758793],
             [ 3.56758793, -3.56758793],
             [ 1.56758793,  0.        ]],

            [[ 1.13517585, -1.13517585],
             [-3.56758793,  2.70276378],
             [-2.70276378,  3.56758793],
             [ 0.        , -1.56758793],
             [ 3.56758793, -3.56758793],
             [ 1.56758793,  0.        ]]]], dtype=np.float64)
            }
]

interpolation_point_data = [
        {
            "ips": np.array([[0.   , 0.   ], [0.   , 0.5  ], [0.   , 1.   ], 
                [0.5  , 0.   ], [0.5  , 0.5  ], [0.5  , 1.   ],
                [1.   , 0.   ], [1.   , 0.5  ], [1.   , 1.   ],
                [0.   , 0.375], [0.   , 0.25 ], [0.   , 0.125],
                [0.125, 0.   ], [0.25 , 0.   ], [0.375, 0.   ],
                [0.375, 0.375], [0.25 , 0.25 ], [0.125, 0.125],
                [0.   , 0.875], [0.   , 0.75 ], [0.   , 0.625],
                [0.125, 0.5  ], [0.25 , 0.5  ], [0.375, 0.5  ],
                [0.375, 0.875], [0.25 , 0.75 ], [0.125, 0.625],
                [0.375, 1.   ], [0.25 , 1.   ], [0.125, 1.   ],
                [0.5  , 0.125], [0.5  , 0.25 ], [0.5  , 0.375],
                [0.625, 0.   ], [0.75 , 0.   ], [0.875, 0.   ],
                [0.875, 0.375], [0.75 , 0.25 ], [0.625, 0.125],
                [0.5  , 0.625], [0.5  , 0.75 ], [0.5  , 0.875],
                [0.625, 0.5  ], [0.75 , 0.5  ], [0.875, 0.5  ],
                [0.875, 0.875], [0.75 , 0.75 ], [0.625, 0.625],
                [0.875, 1.   ], [0.75 , 1.   ], [0.625, 1.   ],
                [1.   , 0.125], [1.   , 0.25 ], [1.   , 0.375],
                [1.   , 0.625], [1.   , 0.75 ], [1.   , 0.875],
                [0.375, 0.125], [0.375, 0.25 ], [0.25 , 0.125],
                [0.875, 0.125], [0.875, 0.25 ], [0.75 , 0.125],
                [0.375, 0.625], [0.375, 0.75 ], [0.25 , 0.625],
                [0.875, 0.625], [0.875, 0.75 ], [0.75 , 0.625],
                [0.125, 0.375], [0.125, 0.25 ], [0.25 , 0.375],
                [0.625, 0.375], [0.625, 0.25 ], [0.75 , 0.375],
                [0.125, 0.875], [0.125, 0.75 ], [0.25 , 0.875],
                [0.625, 0.875], [0.625, 0.75 ], [0.75 , 0.875]], dtype=np.float64),
            "cip": np.array([[ 3, 30, 14, 31, 57, 13, 32, 58, 59, 12,  4, 15, 16, 17,  0],
                [ 6, 51, 35, 52, 60, 34, 53, 61, 62, 33,  7, 36, 37, 38,  3],
                [ 4, 39, 23, 40, 63, 22, 41, 64, 65, 21,  5, 24, 25, 26,  1],
                [ 7, 54, 44, 55, 66, 43, 56, 67, 68, 42,  8, 45, 46, 47,  4],
                [ 1,  9, 21, 10, 69, 22, 11, 70, 71, 23,  0, 17, 16, 15,  4],
                [ 4, 32, 42, 31, 72, 43, 30, 73, 74, 44,  3, 38, 37, 36,  7],
                [ 2, 18, 29, 19, 75, 28, 20, 76, 77, 27,  1, 26, 25, 24,  5],
                [ 5, 41, 50, 40, 78, 49, 39, 79, 80, 48,  4, 47, 46, 45,  8]], dtype=np.int32),
            "fip": np.array([[ 1,  9, 10, 11,  0], [ 0, 12, 13, 14,  3], 
                [ 4, 15, 16, 17,  0], [ 2, 18, 19, 20,  1],
                [ 1, 21, 22, 23,  4], [ 5, 24, 25, 26,  1],
                [ 5, 27, 28, 29,  2], [ 3, 30, 31, 32,  4],
                [ 3, 33, 34, 35,  6], [ 7, 36, 37, 38,  3],
                [ 4, 39, 40, 41,  5], [ 4, 42, 43, 44,  7],
                [ 8, 45, 46, 47,  4], [ 8, 48, 49, 50,  5],
                [ 6, 51, 52, 53,  7], [ 7, 54, 55, 56,  8]], dtype=np.int32)
            }
] 

uniform_refine_data = [
        {
            "node": np.array([[0.  , 0.  ], [1.  , 0.  ], [0.  , 1.  ],
                [0.5 , 0.  ], [0.  , 0.5 ], [0.5 , 0.5 ], [0.25, 0.  ],
                [0.  , 0.25], [0.75, 0.  ], [0.75, 0.25], [0.  , 0.75],
                [0.25, 0.75], [0.25, 0.25], [0.5 , 0.25], [0.25, 0.5 ]], dtype=np.float64),
            "cell": np.array([[ 0,  6,  7], [ 3,  8, 13], [ 4, 14, 10],
                [ 5, 14, 13], [ 6,  3, 12], [ 8,  1,  9], [14,  5, 11], [14,  4, 12],
                [ 7, 12,  4], [13,  9,  5], [10, 11,  2], [13, 12,  3],
                [12,  7,  6], [ 9, 13,  8], [11, 10, 14], [12, 13, 14]], dtype=np.int32),
            "face2cell": np.array([[ 0,  0,  2,  2], [ 0,  0,  1,  1], [ 5,  5,  2,  2],
                [ 5,  5,  0,  0], [10, 10,  1,  1], [10, 10,  0,  0], [ 4,  4,  2,  2], 
                [ 1,  1,  2,  2], [ 4, 11,  0,  0], [ 1, 11,  1,  1], [ 8,  8,  1,  1], 
                [ 2,  2,  1,  1], [ 7,  8,  0,  0], [ 2,  7,  2,  2], [ 9,  9,  0,  0],
                [ 6,  6,  0,  0], [ 3,  9,  1,  1], [ 3,  6,  2,  2], [ 0, 12,  0,  0],
                [ 4, 12,  1,  1], [ 8, 12,  2,  2], [ 5, 13,  1,  1], [ 1, 13,  0,  0],
                [ 9, 13,  2,  2], [10, 14,  2,  2], [ 2, 14,  0,  0], [ 6, 14,  1,  1],
                [11, 15,  2,  2], [ 7, 15,  1,  1], [ 3, 15,  0,  0]], dtype=np.int32),
            "cell2edge": np.array([[18,  1,  0], [22,  9,  7], [25, 11, 13],
                [29, 16, 17], [ 8, 19,  6], [ 3, 21,  2], [15, 26, 17], 
                [12, 28, 13], [12, 10, 20], [14, 16, 23], [ 5,  4, 24], [ 8,  9, 27],
                [18, 19, 20], [22, 21, 23], [25, 26, 24], [29, 28, 27]], dtype=np.int32)
            }
]

jacobian_matrix_data = [
        {"jacobian_matrix": 
        np.array([[[ 0. , -0.5],
        [ 0.5,  0. ]],

       [[ 0. , -0.5],
        [ 0.5,  0. ]],

       [[ 0. , -0.5],
        [ 0.5,  0. ]],

       [[ 0. , -0.5],
        [ 0.5,  0. ]],

       [[ 0. ,  0.5],
        [-0.5,  0. ]],

       [[ 0. ,  0.5],
        [-0.5,  0. ]],

       [[ 0. ,  0.5],
        [-0.5,  0. ]],

       [[ 0. ,  0.5],
        [-0.5,  0. ]]], dtype=np.float64)
            }
]

from_unit_sphere_surface_data= [
            {
                "node": np.array([[ 0.        ,  0.85065081,  0.52573111],
                   [ 0.        ,  0.85065081, -0.52573111],
                   [ 0.85065081,  0.52573111,  0.        ],
                   [ 0.85065081, -0.52573111,  0.        ],
                   [ 0.        , -0.85065081, -0.52573111],
                   [ 0.        , -0.85065081,  0.52573111],
                   [ 0.52573111,  0.        ,  0.85065081],
                   [-0.52573111,  0.        ,  0.85065081],
                   [ 0.52573111,  0.        , -0.85065081],
                   [-0.52573111,  0.        , -0.85065081],
                   [-0.85065081,  0.52573111,  0.        ],
                   [-0.85065081, -0.52573111,  0.        ]], dtype=np.float64),
                "edge": np.array([[ 1,  0], [ 2,  0], [ 0,  6], [ 0,  7], [10,  0],
                   [ 1,  2], [ 8,  1], [ 1,  9], [ 1, 10], [ 3,  2], [ 6,  2],
                   [ 8,  2], [ 3,  4], [ 5,  3], [ 6,  3],  [3,  8], [ 5,  4],
                   [ 4,  8], [ 9,  4], [11,  4], [ 6,  5], [ 7,  5], [ 5, 11],
                   [ 6,  7], [ 7, 10], [11,  7], [ 8,  9], [ 9, 10], [11,  9], [10, 11]], dtype=np.int32),
                "cell": np.array([[ 6,  2,  0], [ 3,  2,  6], [ 5,  3,  6],
                   [ 5,  6,  7], [ 6,  0,  7], [ 3,  8,  2], [ 2,  8,  1], 
                   [ 2,  1,  0], [ 0,  1, 10], [ 1,  9, 10], [ 8,  9,  1], 
                   [ 4,  8,  3], [ 4,  3,  5], [ 4,  5, 11], [ 7, 10, 11],
                   [ 0, 10,  7], [ 4, 11,  9], [ 8,  4,  9], [ 5,  7, 11], 
                   [10,  9, 11]], dtype=np.int32),
                "face2cell": np.array([[ 7,  8,  0,  2],
                   [ 0,  7,  0,  1],
                   [ 0,  4,  1,  2],
                   [ 4, 15,  0,  1],
                   [ 8, 15,  1,  2],
                   [ 6,  7,  1,  2],
                   [ 6, 10,  0,  1],
                   [ 9, 10,  2,  0],
                   [ 8,  9,  0,  1],
                   [ 1,  5,  2,  1],
                   [ 0,  1,  2,  0],
                   [ 5,  6,  0,  2],
                   [11, 12,  1,  2],
                   [ 2, 12,  2,  0],
                   [ 1,  2,  1,  0],
                   [ 5, 11,  2,  0],
                   [12, 13,  1,  2],
                   [11, 17,  2,  2],
                   [16, 17,  1,  0],
                   [13, 16,  1,  2],
                   [ 2,  3,  1,  2],
                   [ 3, 18,  1,  2],
                   [13, 18,  0,  1],
                   [ 3,  4,  0,  1],
                   [14, 15,  2,  0],
                   [14, 18,  1,  0],
                   [10, 17,  2,  1],
                   [ 9, 19,  0,  2],
                   [16, 19,  0,  0],
                   [14, 19,  0,  1]], dtype=np.int32),
                "NN": 12,
                "NE": 30,
                "NF": 30,
                "NC": 20
            }
]

ellipsoid_surface_data= [
            {
                "node": np.array([[ 7.07106781e-01,  0.00000000e+00,  7.07106781e-01],
               [ 5.72061403e-01,  4.15626938e-01,  7.07106781e-01],
               [ 2.18508012e-01,  6.72498512e-01,  7.07106781e-01],
               [-2.18508012e-01,  6.72498512e-01,  7.07106781e-01],
               [-5.72061403e-01,  4.15626938e-01,  7.07106781e-01],
               [-7.07106781e-01,  8.65956056e-17,  7.07106781e-01],
               [-5.72061403e-01, -4.15626938e-01,  7.07106781e-01],
               [-2.18508012e-01, -6.72498512e-01,  7.07106781e-01],
               [ 2.18508012e-01, -6.72498512e-01,  7.07106781e-01],
               [ 5.72061403e-01, -4.15626938e-01,  7.07106781e-01],
               [ 8.09016994e-01,  0.00000000e+00,  5.87785252e-01],
               [ 6.54508497e-01,  4.75528258e-01,  5.87785252e-01],
               [ 2.50000000e-01,  7.69420884e-01,  5.87785252e-01],
               [-2.50000000e-01,  7.69420884e-01,  5.87785252e-01],
               [-6.54508497e-01,  4.75528258e-01,  5.87785252e-01],
               [-8.09016994e-01,  9.90760073e-17,  5.87785252e-01],
               [-6.54508497e-01, -4.75528258e-01,  5.87785252e-01],
               [-2.50000000e-01, -7.69420884e-01,  5.87785252e-01],
               [ 2.50000000e-01, -7.69420884e-01,  5.87785252e-01],
               [ 6.54508497e-01, -4.75528258e-01,  5.87785252e-01],
               [ 8.91006524e-01,  0.00000000e+00,  4.53990500e-01],
               [ 7.20839420e-01,  5.23720495e-01,  4.53990500e-01],
               [ 2.75336158e-01,  8.47397561e-01,  4.53990500e-01],
               [-2.75336158e-01,  8.47397561e-01,  4.53990500e-01],
               [-7.20839420e-01,  5.23720495e-01,  4.53990500e-01],
               [-8.91006524e-01,  1.09116829e-16,  4.53990500e-01],
               [-7.20839420e-01, -5.23720495e-01,  4.53990500e-01],
               [-2.75336158e-01, -8.47397561e-01,  4.53990500e-01],
               [ 2.75336158e-01, -8.47397561e-01,  4.53990500e-01],
               [ 7.20839420e-01, -5.23720495e-01,  4.53990500e-01],
               [ 9.51056516e-01,  0.00000000e+00,  3.09016994e-01],
               [ 7.69420884e-01,  5.59016994e-01,  3.09016994e-01],
               [ 2.93892626e-01,  9.04508497e-01,  3.09016994e-01],
               [-2.93892626e-01,  9.04508497e-01,  3.09016994e-01],
               [-7.69420884e-01,  5.59016994e-01,  3.09016994e-01],
               [-9.51056516e-01,  1.16470832e-16,  3.09016994e-01],
               [-7.69420884e-01, -5.59016994e-01,  3.09016994e-01],
               [-2.93892626e-01, -9.04508497e-01,  3.09016994e-01],
               [ 2.93892626e-01, -9.04508497e-01,  3.09016994e-01],
               [ 7.69420884e-01, -5.59016994e-01,  3.09016994e-01],
               [ 9.87688341e-01,  0.00000000e+00,  1.56434465e-01],
               [ 7.99056653e-01,  5.80548640e-01,  1.56434465e-01],
               [ 3.05212482e-01,  9.39347432e-01,  1.56434465e-01],
               [-3.05212482e-01,  9.39347432e-01,  1.56434465e-01],
               [-7.99056653e-01,  5.80548640e-01,  1.56434465e-01],
               [-9.87688341e-01,  1.20956936e-16,  1.56434465e-01],
               [-7.99056653e-01, -5.80548640e-01,  1.56434465e-01],
               [-3.05212482e-01, -9.39347432e-01,  1.56434465e-01],
               [ 3.05212482e-01, -9.39347432e-01,  1.56434465e-01],
               [ 7.99056653e-01, -5.80548640e-01,  1.56434465e-01],
               [ 1.00000000e+00,  0.00000000e+00,  6.12323400e-17],
               [ 8.09016994e-01,  5.87785252e-01,  6.12323400e-17],
               [ 3.09016994e-01,  9.51056516e-01,  6.12323400e-17],
               [-3.09016994e-01,  9.51056516e-01,  6.12323400e-17],
               [-8.09016994e-01,  5.87785252e-01,  6.12323400e-17],
               [-1.00000000e+00,  1.22464680e-16,  6.12323400e-17],
               [-8.09016994e-01, -5.87785252e-01,  6.12323400e-17],
               [-3.09016994e-01, -9.51056516e-01,  6.12323400e-17],
               [ 3.09016994e-01, -9.51056516e-01,  6.12323400e-17],
               [ 8.09016994e-01, -5.87785252e-01,  6.12323400e-17],
               [ 9.87688341e-01,  0.00000000e+00, -1.56434465e-01],
               [ 7.99056653e-01,  5.80548640e-01, -1.56434465e-01],
               [ 3.05212482e-01,  9.39347432e-01, -1.56434465e-01],
               [-3.05212482e-01,  9.39347432e-01, -1.56434465e-01],
               [-7.99056653e-01,  5.80548640e-01, -1.56434465e-01],
               [-9.87688341e-01,  1.20956936e-16, -1.56434465e-01],
               [-7.99056653e-01, -5.80548640e-01, -1.56434465e-01],
               [-3.05212482e-01, -9.39347432e-01, -1.56434465e-01],
               [ 3.05212482e-01, -9.39347432e-01, -1.56434465e-01],
               [ 7.99056653e-01, -5.80548640e-01, -1.56434465e-01],
               [ 9.51056516e-01,  0.00000000e+00, -3.09016994e-01],
               [ 7.69420884e-01,  5.59016994e-01, -3.09016994e-01],
               [ 2.93892626e-01,  9.04508497e-01, -3.09016994e-01],
               [-2.93892626e-01,  9.04508497e-01, -3.09016994e-01],
               [-7.69420884e-01,  5.59016994e-01, -3.09016994e-01],
               [-9.51056516e-01,  1.16470832e-16, -3.09016994e-01],
               [-7.69420884e-01, -5.59016994e-01, -3.09016994e-01],
               [-2.93892626e-01, -9.04508497e-01, -3.09016994e-01],
               [ 2.93892626e-01, -9.04508497e-01, -3.09016994e-01],
               [ 7.69420884e-01, -5.59016994e-01, -3.09016994e-01],
               [ 8.91006524e-01,  0.00000000e+00, -4.53990500e-01],
               [ 7.20839420e-01,  5.23720495e-01, -4.53990500e-01],
               [ 2.75336158e-01,  8.47397561e-01, -4.53990500e-01],
               [-2.75336158e-01,  8.47397561e-01, -4.53990500e-01],
               [-7.20839420e-01,  5.23720495e-01, -4.53990500e-01],
               [-8.91006524e-01,  1.09116829e-16, -4.53990500e-01],
               [-7.20839420e-01, -5.23720495e-01, -4.53990500e-01],
               [-2.75336158e-01, -8.47397561e-01, -4.53990500e-01],
               [ 2.75336158e-01, -8.47397561e-01, -4.53990500e-01],
               [ 7.20839420e-01, -5.23720495e-01, -4.53990500e-01],
               [ 8.09016994e-01,  0.00000000e+00, -5.87785252e-01],
               [ 6.54508497e-01,  4.75528258e-01, -5.87785252e-01],
               [ 2.50000000e-01,  7.69420884e-01, -5.87785252e-01],
               [-2.50000000e-01,  7.69420884e-01, -5.87785252e-01],
               [-6.54508497e-01,  4.75528258e-01, -5.87785252e-01],
               [-8.09016994e-01,  9.90760073e-17, -5.87785252e-01],
               [-6.54508497e-01, -4.75528258e-01, -5.87785252e-01],
               [-2.50000000e-01, -7.69420884e-01, -5.87785252e-01],
               [ 2.50000000e-01, -7.69420884e-01, -5.87785252e-01],
               [ 6.54508497e-01, -4.75528258e-01, -5.87785252e-01],
               [ 7.07106781e-01,  0.00000000e+00, -7.07106781e-01],
               [ 5.72061403e-01,  4.15626938e-01, -7.07106781e-01],
               [ 2.18508012e-01,  6.72498512e-01, -7.07106781e-01],
               [-2.18508012e-01,  6.72498512e-01, -7.07106781e-01],
               [-5.72061403e-01,  4.15626938e-01, -7.07106781e-01],
               [-7.07106781e-01,  8.65956056e-17, -7.07106781e-01],
               [-5.72061403e-01, -4.15626938e-01, -7.07106781e-01],
               [-2.18508012e-01, -6.72498512e-01, -7.07106781e-01],
               [ 2.18508012e-01, -6.72498512e-01, -7.07106781e-01],
               [ 5.72061403e-01, -4.15626938e-01, -7.07106781e-01]], dtype=np.float64),
                "cell": np.array([[ 10,  11,   0], [  1,   0,  11], [ 20,  21,  10],
               [ 11,  10,  21], [ 30,  31,  20], [ 21,  20,  31], [ 40,  41,  30],
               [ 31,  30,  41], [ 50,  51,  40], [ 41,  40,  51], [ 60,  61,  50],
               [ 51,  50,  61], [ 70,  71,  60], [ 61,  60,  71], [ 80,  81,  70],
               [ 71,  70,  81], [ 90,  91,  80], [ 81,  80,  91], [100, 101,  90],
               [ 91,  90, 101], [ 11,  12,   1], [  2,   1,  12], [ 21,  22,  11],
               [ 12,  11,  22], [ 31,  32,  21], [ 22,  21,  32], [ 41,  42,  31],
               [ 32,  31,  42], [ 51,  52,  41], [ 42,  41,  52], [ 61,  62,  51],
               [ 52,  51,  62], [ 71,  72,  61], [ 62,  61,  72], [ 81,  82,  71],
               [ 72,  71,  82], [ 91,  92,  81], [ 82,  81,  92], [101, 102,  91],
               [ 92,  91, 102], [ 12,  13,   2], [  3,   2,  13], [ 22,  23,  12],
               [ 13,  12,  23], [ 32,  33,  22], [ 23,  22,  33], [ 42,  43,  32],
               [ 33,  32,  43], [ 52,  53,  42], [ 43,  42,  53], [ 62,  63,  52],
               [ 53,  52,  63], [ 72,  73,  62], [ 63,  62,  73], [ 82,  83,  72],
               [ 73,  72,  83], [ 92,  93,  82], [ 83,  82,  93], [102, 103,  92],
               [ 93,  92, 103], [ 13,  14,   3], [  4,   3,  14], [ 23,  24,  13],
               [ 14,  13,  24], [ 33,  34,  23], [ 24,  23,  34], [ 43,  44,  33],
               [ 34,  33,  44], [ 53,  54,  43], [ 44,  43,  54], [ 63,  64,  53],
               [ 54,  53,  64], [ 73,  74,  63], [ 64,  63,  74], [ 83,  84,  73],
               [ 74,  73,  84], [ 93,  94,  83], [ 84,  83,  94], [103, 104,  93],
               [ 94,  93, 104], [ 14,  15,   4], [  5,   4,  15], [ 24,  25,  14],
               [ 15,  14,  25], [ 34,  35,  24], [ 25,  24,  35], [ 44,  45,  34],
               [ 35,  34,  45], [ 54,  55,  44], [ 45,  44,  55], [ 64,  65,  54],
               [ 55,  54,  65], [ 74,  75,  64], [ 65,  64,  75], [ 84,  85,  74],
               [ 75,  74,  85], [ 94,  95,  84], [ 85,  84,  95], [104, 105,  94],
               [ 95,  94, 105], [ 15,  16,   5], [  6,   5,  16], [ 25,  26,  15], 
               [ 16,  15,  26], [ 35,  36,  25], [ 26,  25,  36], [ 45,  46,  35],
               [ 36,  35,  46], [ 55,  56,  45], [ 46,  45,  56], [ 65,  66,  55],
               [ 56,  55,  66], [ 75,  76,  65], [ 66,  65,  76], [ 85,  86,  75],
               [ 76,  75,  86], [ 95,  96,  85], [ 86,  85,  96], [105, 106,  95],
               [ 96,  95, 106], [ 16,  17,   6], [  7,   6,  17], [ 26,  27,  16],
               [ 17,  16,  27], [ 36,  37,  26], [ 27,  26,  37], [ 46,  47,  36],
               [ 37,  36,  47], [ 56,  57,  46], [ 47,  46,  57], [ 66,  67,  56],
               [ 57,  56,  67], [ 76,  77,  66], [ 67,  66,  77], [ 86,  87,  76],
               [ 77,  76,  87], [ 96,  97,  86], [ 87,  86,  97], [106, 107,  96],
               [ 97,  96, 107], [ 17,  18,   7], [  8,   7,  18], [ 27,  28,  17],
               [ 18,  17,  28], [ 37,  38,  27], [ 28,  27,  38], [ 47,  48,  37],
               [ 38,  37,  48], [ 57,  58,  47], [ 48,  47,  58], [ 67,  68,  57],
               [ 58,  57,  68], [ 77,  78,  67], [ 68,  67,  78], [ 87,  88,  77],
               [ 78,  77,  88], [ 97,  98,  87], [ 88,  87,  98], [107, 108,  97],
               [ 98,  97, 108], [ 18,  19,   8], [  9,   8,  19], [ 28,  29,  18],
               [ 19,  18,  29], [ 38,  39,  28], [ 29,  28,  39], [ 48,  49,  38],
               [ 39,  38,  49], [ 58,  59,  48], [ 49,  48,  59], [ 68,  69,  58],
               [ 59,  58,  69], [ 78,  79,  68], [ 69,  68,  79], [ 88,  89,  78],
               [ 79,  78,  89], [ 98,  99,  88], [ 89,  88,  99], [108, 109,  98],
               [ 99,  98, 109], [ 19,  10,   9], [  0,   9,  10], [ 29,  20,  19],
               [ 10,  19,  20], [ 39,  30,  29], [ 20,  29,  30], [ 49,  40,  39],
               [ 30,  39,  40], [ 59,  50,  49], [ 40,  49,  50], [ 69,  60,  59],
               [ 50,  59,  60], [ 79,  70,  69], [ 60,  69,  70], [ 89,  80,  79],
               [ 70,  79,  80], [ 99,  90,  89], [ 80,  89,  90], [109, 100,  99], [ 90,  99, 100]], dtype=np.int32), 
                "NN": 110,
                "NE": 310,
                "NF": 310,
                "NC": 200
            }
]
