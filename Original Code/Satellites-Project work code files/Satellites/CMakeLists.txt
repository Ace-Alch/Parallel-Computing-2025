CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
project(Satellites)

add_executable(parallel parallel.c)


# Here is an example syntax how to add compiler options to your build process
# See the project work document on compiler flag syntax on Linux and Windows

#target_compile_options(parallel PRIVATE "/O2" "/fp:fast" "/Qvec-report:2")




# Prerequisite for enabling OpenMP on macOS.
# Assumes libomp is installed through Homebrew, feel free to modify.
# UNCOMMENT FOLLOWING CONDITIONAL ONLY IF USING macOS
# if(APPLE)
#   execute_process(
#       COMMAND brew --prefix
#       OUTPUT_VARIABLE HOMEBREW_PREFIX
#       OUTPUT_STRIP_TRAILING_WHITESPACE
#   )
#   set(ENV{OpenMP_ROOT} "${HOMEBREW_PREFIX}/opt/libomp")
#   set(ENV{LDFLAGS} "$LDFLAGS -L${OpenMP_ROOT}/lib")
#   set(ENV{CPPFLAGS} "-I${OpenMP_ROOT}/include")
# endif()
#
# UNCOMMENT THESE TO ENABLE OPENMP
# find_package(OpenMP REQUIRED)
# target_link_libraries(parallel OpenMP::OpenMP_C)


# UNCOMMENT THESE TO ENABLE OPENCL
# This will also copy the kernel source file parallel.cl (created by you) to the build directory
# The copying command is unfortunately not perfect, as it doesn't redo the copy if you only edit
# the parallel.cl, but leave the parallel.c untouched.
# Because of this, you might need to force 'Rebuild All' to ensure kernel code updates propagate
# to the build directory. If you know enough CMake magic, feel free to fix this and let me know.
# Note, that the build seems to fail (at least on Visual Studio) if parallel.cl does not exist.
#
# find_package(OpenCL REQUIRED)
# target_include_directories(parallel PRIVATE ${OpenCL_INCLUDE_DIRS})
# target_link_libraries(parallel ${OpenCL_LIBRARIES})
# add_custom_command(
#     TARGET parallel POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${CMAKE_SOURCE_DIR}/parallel.cl"
#     $<TARGET_FILE_DIR:parallel>
#     VERBATIM)


# Find and link SDL2
if (WIN32)
    set(SDL2_DIR "${CMAKE_SOURCE_DIR}/SDL2-2.32.10/cmake")
endif()
find_package(SDL2 REQUIRED)
target_include_directories(parallel PRIVATE ${SDL2_INCLUDE_DIR})
target_link_libraries(parallel ${SDL2_LIBRARIES})

if (WIN32)
# copy the sdl2 .dll file to the same folder as the executable
add_custom_command(
    TARGET parallel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL2::SDL2>
    $<TARGET_FILE_DIR:parallel>
    VERBATIM)
else()
    # Math library shouldn't be linked on Windows, but must be linked on Linux
    target_link_libraries(parallel m)
endif()
