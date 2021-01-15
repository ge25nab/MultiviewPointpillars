from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_aware_pool3d',
    ext_modules=[
        CUDAExtension('roiaware_pool3d_cuda', [
            'src/roiaware_pool3d_kernel.cu',
            'src/roiaware_pool3d.cpp',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
