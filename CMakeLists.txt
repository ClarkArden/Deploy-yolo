cmake_minimum_required(VERSION 3.10)

project(YOLO_TensorRT_Deployment)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)  # 89：Ada Lovelace 架构（如 RTX 40 系列）


set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# 查找CUDA
find_package(CUDA REQUIRED)

# 查找TensorRT
set(TENSORRT_ROOT "/home/clark/library/TensorRT-8.6.1.6")  # 替换为你的TensorRT安装路径
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARY_DIRS "${TENSORRT_ROOT}/lib")

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${CUDA_INCLUDE_DIRS}
                    ${TENSORRT_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)


file(GLOB SRC_FILES "src/cpp/*.cpp")
file(GLOB CUDA_FILES "src/cpp/*.cu")  # 查找所有.cu文件


link_directories(${TENSORRT_LIBRARY_DIRS})
# 添加可执行文件
add_executable(yolo_tensorrt ${SRC_FILES} ${CUDA_FILES})
# 链接库
target_link_libraries(yolo_tensorrt ${CUDA_LIBRARIES}
                                    ${OpenCV_LIBRARIES}
                                    nvinfer nvinfer_plugin nvonnxparser
                                    stdc++fs)



# 添加CUDA支持
set_property(TARGET yolo_tensorrt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET yolo_tensorrt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# 添加模型文件到构建目录
# configure_file(${PROJECT_SOURCE_DIR}/models/yolo.onnx ${PROJECT_BINARY_DIR}/models/yolo.onnx COPYONLY)