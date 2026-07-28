"""
GPU / Numba JIT Матричный Ускоритель (GpuMatrixAccelerator)
Параллельное векторизованное косинусное сходство по матрицам нейронов высочайшей скорости
"""

import numpy as np

try:
    import torch
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_CUDA_AVAILABLE = False

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
    """Ускоритель поиска сходства по всей оперативной памяти нейронов (PyTorch CUDA / Numba JIT)"""

    def __init__(self):
        self.use_torch_cuda = TORCH_CUDA_AVAILABLE
        self.use_numba = NUMBA_AVAILABLE

    def compute_similarities(self, query_vector, neuron_matrix):
        """Ультра-быстрое вычисление косинусного сходства на GPU CUDA или Numba JIT"""
        if len(neuron_matrix) == 0:
            return np.array([], dtype=np.float32)

        query = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray(neuron_matrix, dtype=np.float32)

        if query.shape[0] != matrix.shape[1]:
            min_dim = min(query.shape[0], matrix.shape[1])
            query = query[:min_dim]
            matrix = matrix[:, :min_dim]

        # 1. Если доступна NVIDIA CUDA через PyTorch
        if self.use_torch_cuda:
            try:
                q_t = torch.tensor(query, device='cuda', dtype=torch.float32)
                m_t = torch.tensor(matrix, device='cuda', dtype=torch.float32)
                dots = torch.matmul(m_t, q_t)
                q_norm = torch.norm(q_t)
                m_norms = torch.norm(m_t, dim=1)
                m_norms[m_norms == 0] = 1e-9
                if q_norm == 0:
                    q_norm = 1e-9
                sims = dots / (m_norms * q_norm)
                return sims.cpu().numpy()
            except Exception:
                pass

        # 2. Высокоскоростной Numba JIT C-ускоритель (SIMD Vectorization)
        return numba_cosine_similarity_matrix(query, matrix)
