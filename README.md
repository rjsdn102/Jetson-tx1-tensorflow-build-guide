# Jetson TX1 tensorflow r0.9 Build

# References
https://stackoverflow.com/questions/39783919/tensorflow-on-nvidia-tx1/

http://www.yuthon.com/2016/12/04/Installation-of-TensorFlow-r0-11-on-TX1/index.html

https://github.com/smajida/Tensorflow-r0.8-on-Jetson-TX1

https://www.slothparadise.com/setup-cuda-7-0-nvidia-jetson-tx1-jetpack-detailed/

build on 2017-08-10

# Environment
```
Ubuntu 16.04 LTS (from Jetpack 2.3.1)
CUDA 8.0
cuDNN 5.1
```
# CUDA install after flashing
```
$ cd ~/cuda-l4t
$ sudo ./cuda-l4t.sh cuda-repo-l4t-8-0-local_8.0.34-1_arm64.deb 8.0 8-0
$ source ~/.bashrc
$ nvcc -V
```
You can check if CUDA 8.0 is installed
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Fri_Jul_15_14:52:12_CDT_2016
Cuda compilation tools, release 8.0, V8.0.33
```
# Install Java
```
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
```
# Install dependences
```
$ sudo apt-get install git zip unzip autoconf automake libtool curl zlib1g-dev maven
$ sudo apt-get install python-numpy swig python-dev python-wheel
```
# Build protobuf
```
$ git clone https://github.com/google/protobuf.git
$ cd protobuf
$ # autogen.sh downloads broken gmock.zip in d5fb408d
$ git checkout master
$ ./autogen.sh
$ git checkout d5fb408d
$ ./configure --prefix=/usr
$ make -j 4
$ sudo make install
$ cd java
$ mvn package
```
# Build Bazel
```
$ git clone https://github.com/bazelbuild/bazel.git
$ cd bazel
$ git checkout 0.2.1
$ cp /usr/bin/protoc third_party/protobuf/protoc-linux-arm32.exe
cp ../protobuf/java/target/protobuf-java-3.0.0-beta-2.jar third_party/protobuf/protobuf-java-3.0.0-beta-1.jar
```
You need to edit a bazel file to recognize aarch64 as ARM
```
--- a/src/main/java/com/google/devtools/build/lib/util/CPU.java
+++ b/src/main/java/com/google/devtools/build/lib/util/CPU.java
@@ -25,7 +25,7 @@ import java.util.Set;
 public enum CPU {
   X86_32("x86_32", ImmutableSet.of("i386", "i486", "i586", "i686", "i786", "x86")),
   X86_64("x86_64", ImmutableSet.of("amd64", "x86_64", "x64")),
-  ARM("arm", ImmutableSet.of("arm", "armv7l")),
+  ARM("arm", ImmutableSet.of("arm", "armv7l", "aarch64")),
   UNKNOWN("unknown", ImmutableSet.<String>of());
```
Now compile
```
$ ./compile.sh
```
and install
```
sudo cp output/bazel /usr/local/bin
```
# Build Tensorflow r0.9
```
$ git clone -b r0.9 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ ./configure
```
And then you can see these sentenses
```
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow
Please specify which gcc nvcc should use as the host compiler. [Default is /usr/bin/gcc]:
Please specify the Cuda SDK version you want to use, e.g. 7.0\. [Leave empty to use system default]:
Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify the Cudnn version you want to use. [Leave empty to use system default]:
Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 5.3
Setting up Cuda include
Setting up Cuda lib
Setting up Cuda bin
Setting up Cuda nvvm
Configuration finished
```

Before you build tensorflow, you have to fix some files.
```
--- a/tensorflow/core/kernels/BUILD
+++ b/tensorflow/core/kernels/BUILD
@@ -985,7 +985,7 @@ tf_kernel_libraries(
         "reduction_ops",
         "segment_reduction_ops",
         "sequence_ops",
-        "sparse_matmul_op",
+        #DC "sparse_matmul_op",
     ],
     deps = [
         ":bounds_check",

--- a/tensorflow/python/BUILD
+++ b/tensorflow/python/BUILD
@@ -1110,7 +1110,7 @@ medium_kernel_test_list = glob([
     "kernel_tests/seq2seq_test.py",
     "kernel_tests/slice_op_test.py",
     "kernel_tests/sparse_ops_test.py",
-    "kernel_tests/sparse_matmul_op_test.py",
+    #DC "kernel_tests/sparse_matmul_op_test.py",
     "kernel_tests/sparse_tensor_dense_matmul_op_test.py",
 ])
 
 --- a/tensorflow/core/kernels/cwise_op_gpu_select.cu.cc
+++ b/tensorflow/core/kernels/cwise_op_gpu_select.cu.cc
@@ -43,8 +43,14 @@ struct BatchSelectFunctor<GPUDevice, T> {
     const int all_but_batch = then_flat_outer_dims.dimension(1);

 #if !defined(EIGEN_HAS_INDEX_LIST)
-    Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
-    Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
+    //DC Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
+    Eigen::array<int, 2> broadcast_dims;
+    broadcast_dims[0] = 1;
+    broadcast_dims[1] = all_but_batch;
+    //DC Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
+    Eigen::Tensor<int, 2>::Dimensions reshape_dims;
+    reshape_dims[0] = batch;
+    reshape_dims[1] = 1;
 #else
     Eigen::IndexList<Eigen::type2index<1>, int> broadcast_dims;
     broadcast_dims.set(1, all_but_batch);

--- a/tensorflow/core/kernels/sparse_tensor_dense_matmul_op_gpu.cu.cc
+++ b/tensorflow/core/kernels/sparse_tensor_dense_matmul_op_gpu.cu.cc
@@ -104,9 +104,17 @@ struct SparseTensorDenseMatMulFunctor<GPUDevice, T, ADJ_A, ADJ_B> {
     int n = (ADJ_B) ? b.dimension(0) : b.dimension(1);

 #if !defined(EIGEN_HAS_INDEX_LIST)
-    Eigen::Tensor<int, 2>::Dimensions matrix_1_by_nnz{{ 1, nnz }};
-    Eigen::array<int, 2> n_by_1{{ n, 1 }};
-    Eigen::array<int, 1> reduce_on_rows{{ 0 }};
+    //DC Eigen::Tensor<int, 2>::Dimensions matrix_1_by_nnz{{ 1, nnz }};
+    Eigen::Tensor<int, 2>::Dimensions matrix_1_by_nnz;
+    matrix_1_by_nnz[0] = 1;
+    matrix_1_by_nnz[1] = nnz;
+    //DC Eigen::array<int, 2> n_by_1{{ n, 1 }};
+    Eigen::array<int, 2> n_by_1;
+    n_by_1[0] = n;
+    n_by_1[1] = 1;
+    //DC Eigen::array<int, 1> reduce_on_rows{{ 0 }};
+    Eigen::array<int, 1> reduce_on_rows;
+    reduce_on_rows[0] = 0;
 #else
     Eigen::IndexList<Eigen::type2index<1>, int> matrix_1_by_nnz;
     matrix_1_by_nnz.set(1, nnz);

--- a/tensorflow/stream_executor/cuda/cuda_blas.cc
+++ b/tensorflow/stream_executor/cuda/cuda_blas.cc
@@ -25,6 +25,12 @@ limitations under the License.
 #define EIGEN_HAS_CUDA_FP16
 #endif

+#if CUDA_VERSION >= 8000
+#define SE_CUDA_DATA_HALF CUDA_R_16F
+#else
+#define SE_CUDA_DATA_HALF CUBLAS_DATA_HALF
+#endif
+
 #include "tensorflow/stream_executor/cuda/cuda_blas.h"

 #include <dlfcn.h>
@@ -1680,10 +1686,10 @@ bool CUDABlas::DoBlasGemm(
   return DoBlasInternal(
       dynload::cublasSgemmEx, stream, true /* = pointer_mode_host */,
       CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
-      CUDAMemory(a), CUBLAS_DATA_HALF, lda,
-      CUDAMemory(b), CUBLAS_DATA_HALF, ldb,
+      CUDAMemory(a), SE_CUDA_DATA_HALF, lda,
+      CUDAMemory(b), SE_CUDA_DATA_HALF, ldb,
       &beta,
-      CUDAMemoryMutable(c), CUBLAS_DATA_HALF, ldc);
+      CUDAMemoryMutable(c), SE_CUDA_DATA_HALF, ldc);
 #else
   LOG(ERROR) << "fp16 sgemm is not implemented in this cuBLAS version "
              << "(need at least CUDA 7.5)";
              
--- a/tensorflow/stream_executor/cuda/cuda_gpu_executor.cc
+++ b/tensorflow/stream_executor/cuda/cuda_gpu_executor.cc
@@ -888,6 +888,9 @@ CudaContext* CUDAExecutor::cuda_context() { return context_; }
 // For anything more complicated/prod-focused than this, you'll likely want to
 // turn to gsys' topology modeling.
 static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
+  // DC - make this clever later. ARM has no NUMA node, just return 0
+  LOG(INFO) << "ARM has no NUMA node, hardcoding to return zero";
+  return 0;
 #if defined(__APPLE__)
   LOG(INFO) << "OS X does not support NUMA - returning NUMA node zero";
   return 0;              
```

Now build tensorflow using bazel
```
~/tensorflow$ bazel build -c opt --local_resources 2048,1.0,1.0 --verbose_failures -s --config=cuda //tensorflow/cc:tutorials_example_trainer
```
If you got the error about 'gcc' change 2048,1.0,1.0 to 2048,0.5,1.0

And then test whether tensorflow built or not
```
~/tensorflow$ bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu
```
You can see the sentenses such as these if the build is successful.
```
000009/000005 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000001 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000009/000009 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
```
Finally, build tensorflow for using on python.
```
~/tensorflow$ bazel build -c opt --local_resources 2048,1.0,1.0 --verbose_failures --config=cuda //tensorflow/tools/pip_package:build_pip_package
~/tensorflow$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
~/tensorflow$ sudo pip install /tmp/tensorflow_pkg/tensorflow-0.9.0-py2-none-any.whl
```
If you got the errors such as about pip or setup, 
```
sudo easy_install pip
```
```
sudo apt-get install python-setuptools
```
Building Finished!!

# Tip : Creating a swap file
```
# Create a swapfile for Ubuntu at the current directory location
fallocate -l *G swapfile
# List out the file
ls -lh swapfile
# Change permissions so that only root can use it
chmod 600 swapfile
# List out the file
ls -lh swapfile
# Set up the Linux swap area
mkswap swapfile
# Now start using the swapfile
sudo swapon swapfile
# Show that it's now being used
swapon -s
```
