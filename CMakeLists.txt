project(wl_cpp)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
find_package(Boost COMPONENTS python REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)

include_directories(
    /usr/include/python2.7
    ${OPENCV_INCLUDE_DIRS}
)

add_executable(wl_cpp
    main.cpp
)

target_link_libraries(wl_cpp
    Boost::python
    boost_numpy
    python2.7
    ${OpenCV_LIBS}
)

