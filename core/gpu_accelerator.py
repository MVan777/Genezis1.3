"""
GPU / Numba JIT Матричный Ускоритель (GpuMatrixAccelerator)
Параллельное векторизованное косинусное сходство по матрицам нейронов высочайшей скорости
"""

import numpy as np

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True)
    def numba_cosine_similarity_matrix(query, matrix):
        """Numba JIT C-компилированный матричный косинусный поиск"""
        dot_products = np.dot(matrix, query)
        query_norm = np.linalg.norm(query)
        matrix_norms = np.sqrt(np.sum(matrix ** 2, axis=1))
        
        # Избегаем деления на ноль
        matrix_norms[matrix_norms == 0] = 1e-9
        if query_norm == 0:
            query_norm = 1e-9

        return dot_products / (matrix_norms * query_norm)
else:
    def numba_cosine_similarity_matrix(query, matrix):
        """Векторизованный NumPy fallback"""
        dot_products = np.dot(matrix, query)
        query_norm = np.linalg.norm(query)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        matrix_norms[matrix_norms == 0] = 1e-9
        if query_norm == 0:
            query_norm = 1e-9

        return dot_products / (matrix_norms * query_norm)

class GpuMatrixAccelerator:
    """Ускоритель поиска сходства по всей оперативной памяти нейронов"""

    def __init__(self):
        self.use_numba = NUMBA_AVAILABLE

    def compute_similarities(self, query_vector, neuron_matrix):
        """Ультра-быстрое вычисление косинусного сходства"""
        if len(neuron_matrix) == 0:
            return np.array([], dtype=np.float32)

        query = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray(neuron_matrix, dtype=np.float32)

        # Выравниваем размерности если нужно
        if query.shape[0] != matrix.shape[1]:
            min_dim = min(query.shape[0], matrix.shape[1])
            query = query[:min_dim]
            matrix = matrix[:, :min_dim]

        return numba_cosine_similarity_matrix(query, matrix)
