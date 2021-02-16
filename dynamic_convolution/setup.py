from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dynamic_convolution',
    ext_modules=[
        CUDAExtension('dynamic_convolution', [
            'dynamic_convolution.cpp',
            'dynamicconv_cuda_forward.cu',
            'dynamicconv_cuda_backward.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })