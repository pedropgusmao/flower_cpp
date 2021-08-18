# flower_cpp

## Requirements
### Libtorch

 Download here (Pre-cxx11 ABI):
https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.0%2Bcu111.zip

Download here (cxx11 ABI):
https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip

### Torchvision
Download here
https://github.com/pytorch/vision#c-api
## Build
Run the following:
```
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="path_to_torchlib;path_to_torchvision" ..
cmake --build . 

```

### Other References:
- (PyTorch-CPP)[https://github.com/prabhuomkar/pytorch-cpp]