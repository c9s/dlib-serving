cmake_minimum_required(VERSION 3.0.12)
# Every project needs a name.  We call this the "examples" project.
project(dlib_training)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

message(${CMAKE_MODULE_PATH})

find_package(Boost 1.45.0 COMPONENTS system filesystem program_options)
find_package(JsonCpp REQUIRED)

# apt-get install libgif-dev libgif7
find_package(GIF REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


# apt-get install libcblas-dev libcblas3
find_package(BLAS)
find_package(LAPACK)
find_package(Gearman REQUIRED)

set(DLIB_LIBS "${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY}")

if(LAPACK_FOUND)
set(DLIB_LIBS "${DLIB_LIBS} ${LAPACK_LIBRARIES}")
endif()

if (BLAS_FOUND)
set(DLIB_LIBS "${DLIB_LIBS} ${BLAS_LIBRARIES}")
endif()

message(${DLIB_LIBS})



if(DLIB_ROOT)
    # Tell cmake we will need dlib.  This command will pull in dlib and compile it
    # into your project.  Note that you don't need to compile or install dlib.  All
    # it needs is the dlib source code folder and it will take care of everything.
    include(${DLIB_ROOT}/dlib/cmake)
else(DLIB_ROOT)
    find_library(DLIB_LIBRARIES
        NAMES dlib
        PATHS
        /opt/local/lib /usr/local/lib /usr/lib)
    find_path(DLIB_INCLUDE_DIRS
        NAMES dlib/config.h
        PATHS /opt/local/include /usr/local/include /usr/include)
    include_directories(${DLIB_INCLUDE_DIRS})

    if(DLIB_LIBRARIES)
        add_library(dlib::dlib INTERFACE IMPORTED)
        set_target_properties(dlib::dlib PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${DLIB_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${DLIB_LIBRARIES}")
        message(${DLIB_LIBRARIES})
        message(${DLIB_INCLUDE_DIRS})
    endif()
endif(DLIB_ROOT)

# find_library(DLIB_LIBRARY_DEBUG NAMES dlib_d HINTS /opt/local/lib /usr/local/lib /usr/lib)

# find_package(OpenCV REQUIRED)
# find_package(GooglePerfTools REQUIRED)
# find_package(GLog REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS} ${Gearman_INCLUDE_DIRS} ${GIF_INCLUDE_DIR} ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${DLIB_INCLUDE_DIRS})

# Since there are a lot of examples I'm going to use a macro to simply this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} dlib::dlib)
endmacro()

# add_example(dnn_face_recognition_linker)


# define the library
# add_library(a3 a.cc b.cc d.cc)


add_library(sp_training_library STATIC sp_training.cc)
target_link_libraries(sp_training_library ${Boost_LIBRARIES} JsonCpp::JsonCpp ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} dlib::dlib)

add_library(resnet_training_library STATIC resnet_training.cc)
target_link_libraries(resnet_training_library  Threads::Threads ${Boost_LIBRARIES} JsonCpp::JsonCpp ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} dlib::dlib)

add_library(face_crop_library STATIC face_crop.cc)
target_link_libraries(face_crop_library Threads::Threads ${Boost_LIBRARIES} JsonCpp::JsonCpp ${GIF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARY} ${BLAS_LIBRARIES} dlib::dlib)

# if(Boost_FOUND AND Gearman_FOUND AND JsonCpp_FOUND)

add_executable(sp-training-worker sp_training_worker.cc)
target_link_libraries(sp-training-worker ${Gearman_LIBS} ${Boost_LIBRARIES} JsonCpp::JsonCpp sp_training_library)

add_executable(sp-training-client sp_training_client.cc)
target_link_libraries(sp-training-client ${Gearman_LIBS} ${Boost_LIBRARIES} JsonCpp::JsonCpp sp_training_library)

add_executable(resnet-training-worker resnet_training_worker.cc)
target_link_libraries(resnet-training-worker ${Gearman_LIBS} ${Boost_LIBRARIES} JsonCpp::JsonCpp resnet_training_library)

add_executable(resnet-training-client resnet_training_client.cc)
target_link_libraries(resnet-training-client ${Gearman_LIBS} ${Boost_LIBRARIES} JsonCpp::JsonCpp)

add_executable(face-crop-worker face_crop_worker.cc)
target_link_libraries(face-crop-worker ${Gearman_LIBS} ${Boost_LIBRARIES} JsonCpp::JsonCpp face_crop_library)

add_executable(face-crop-client face_crop_client.cc)
target_link_libraries(face-crop-client ${Gearman_LIBS} ${Boost_LIBRARIES} JsonCpp::JsonCpp face_crop_library)
