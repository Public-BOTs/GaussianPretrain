from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='bev_pool',
    ext_modules=[
        CUDAExtension('bev_pool_ext', [
            'src/bev_pool_cuda.cu',
            'src/bev_pool.cpp'
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
