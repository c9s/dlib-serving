cmake_minimum_required(VERSION 3.5)
project(dlibinfer)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH /opt/local)
set(CMAKE_CXX_STANDARD 11)

# Configure boost
# =========================================
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.45.0 COMPONENTS system filesystem program_options)

# Configure pthread
# =========================================
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Configure dlib packages...
# =========================================
# apt-get install libgif-dev libgif7
find_package(GIF REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
# apt-get install libcblas-dev libcblas3
find_package(BLAS)
find_package(LAPACK)

if(DLIB_ROOT)
    add_subdirectory(${DLIB_ROOT})
else(DLIB_ROOT)
    find_package(dlib REQUIRED)
endif(DLIB_ROOT)

find_package(Protobuf REQUIRED)
find_package(GRPC REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})

# the protobuf_generate_cpp-generated pb files are located at CMAKE_CURRENT_BINARY_DIR
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/protos)
set(PROTO_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/protos/serving.pb.cc")
set(PROTO_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/protos/serving.pb.h")

set(GRPC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/protos/serving.grpc.pb.cc")
set(GRPC_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/protos/serving.grpc.pb.h")

# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/serving.proto)
# protobuf_generate_python(PROTO_PY ${CMAKE_CURRENT_SOURCE_DIR}/serving.proto)
# grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/protos ${CMAKE_CURRENT_SOURCE_DIR}/protos/serving.proto)

link_libraries(${GRPC_GRPC++_LIBRARY} ${GRPC_LIBRARY} ${PROTOBUF_LIBRARY})
include_directories(${GRPC_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS} ${GIF_INCLUDE_DIR} ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${DLIB_INCLUDE_DIRS})

# define the library
# add_library(foo a.cc b.cc d.cc)

add_executable(train-shape-predictor train_shape_predictor.cc)
target_link_libraries(train-shape-predictor ${Boost_LIBRARIES} dlib::dlib)

add_executable(shape-detection-server ${PROTO_HDRS} ${PROTO_SRCS} ${GRPC_HDRS} ${GRPC_SRCS} jpeg_loader.cc shape_detection_service.cc shape_detection_server.cc)
target_link_libraries(shape-detection-server ${Boost_LIBRARIES} ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${PROTOBUF_LIBRARY} dlib::dlib)

add_executable(face-detection-server ${PROTO_HDRS} ${PROTO_SRCS} ${GRPC_HDRS} ${GRPC_SRCS} jpeg_loader.cc face_detection_service.cc face_detection_server.cc)
target_link_libraries(face-detection-server ${Boost_LIBRARIES} ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} ${PROTOBUF_LIBRARY} dlib::dlib)

add_executable(face-detection-client ${PROTO_HDRS} ${PROTO_SRCS} ${GRPC_HDRS} ${GRPC_SRCS} jpeg_loader.cc face_detection_client.cc)
target_link_libraries(face-detection-client ${Boost_LIBRARIES} ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} ${PROTOBUF_LIBRARY} dlib::dlib)
