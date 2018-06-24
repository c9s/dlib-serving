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
find_package(JsonCpp REQUIRED)

# apt-get install libgif-dev libgif7
find_package(GIF REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

# Configure pthread
# =========================================
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


# Configure dlib packages...
# =========================================
# apt-get install libcblas-dev libcblas3
find_package(BLAS)
find_package(LAPACK)

set(DLIB_LIBS "${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY}")
if(LAPACK_FOUND)
set(DLIB_LIBS "${DLIB_LIBS} ${LAPACK_LIBRARIES}")
endif()
if (BLAS_FOUND)
set(DLIB_LIBS "${DLIB_LIBS} ${BLAS_LIBRARIES}")
endif()
# message("DLIB_LIBS: ${DLIB_LIBS}")


find_package(dlib REQUIRED)
find_package(Protobuf REQUIRED)
find_package(GRPC REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# the protobuf_generate_cpp-generated pb files are located at CMAKE_CURRENT_BINARY_DIR
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS inference.proto)
protobuf_generate_python(PROTO_PY inference.proto)
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${CMAKE_CURRENT_BINARY_DIR} inference.proto)

link_libraries(${GRPC_GRPC++_LIBRARY} ${GRPC_LIBRARY} ${PROTOBUF_LIBRARY})
include_directories(${GRPC_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS} ${GIF_INCLUDE_DIR} ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${DLIB_INCLUDE_DIRS})

# define the library
# add_library(foo a.cc b.cc d.cc)

add_executable(train-shape-predictor train_shape_predictor.cc)
target_link_libraries(train-shape-predictor ${Boost_LIBRARIES} JsonCpp::JsonCpp ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} dlib::dlib)

add_executable(shape_detection_server ${PROTO_HDRS} ${PROTO_SRCS} ${GRPC_HDRS} ${GRPC_SRCS} jpeg_loader.cc shape_detection_service.cc shape_detection_server.cc)
target_link_libraries(shape_detection_server ${Boost_LIBRARIES} JsonCpp::JsonCpp ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} ${PROTOBUF_LIBRARY} dlib::dlib)

add_executable(face_detection_server ${PROTO_HDRS} ${PROTO_SRCS} ${GRPC_HDRS} ${GRPC_SRCS} jpeg_loader.cc face_detection_service.cc face_detection_server.cc)
target_link_libraries(face_detection_server ${Boost_LIBRARIES} JsonCpp::JsonCpp ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} ${PROTOBUF_LIBRARY} dlib::dlib)

add_executable(face_detection_client ${PROTO_HDRS} ${PROTO_SRCS} ${GRPC_HDRS} ${GRPC_SRCS} jpeg_loader.cc face_detection_client.cc)
target_link_libraries(face_detection_client ${Boost_LIBRARIES} JsonCpp::JsonCpp ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} ${PROTOBUF_LIBRARY} dlib::dlib)
