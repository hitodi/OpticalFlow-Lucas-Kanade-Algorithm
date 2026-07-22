cmake_minimum_required(VERSION 3.16)

project(LucasKanadeOpticalFlow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(lucas_kanade_optical_flow
    test/test.cpp
)

target_link_libraries(lucas_kanade_optical_flow
    PRIVATE
        ${OpenCV_LIBS}
)

target_include_directories(lucas_kanade_optical_flow
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

if (MSVC)
    target_compile_options(lucas_kanade_optical_flow PRIVATE /W4)
else()
    target_compile_options(lucas_kanade_optical_flow PRIVATE -Wall -Wextra -Wpedantic)
endif()
