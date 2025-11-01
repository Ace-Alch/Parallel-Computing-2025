CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(VectorAdd)

add_executable(VectorAdd VectorAdd.c)

find_package(OpenCL REQUIRED)
target_include_directories(VectorAdd PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(VectorAdd ${OpenCL_LIBRARIES})

add_custom_command(
    TARGET VectorAdd POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/VectorAdd.cl"
    $<TARGET_FILE_DIR:VectorAdd>
    VERBATIM)

