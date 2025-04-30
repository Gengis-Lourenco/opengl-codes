# ==============================================================================
# Project   : Salami-OpenGL Mesh Movie
# File      : CMakeLists.txt
# Purpose   : Configure a cross-platform build (Linux & macOS) for:
#             - filter (filtershader.cpp)
#             - salami_lightning (Salami_lightning.cpp)
#             - salami_window (Salami_movie_glwindow.cpp)
#
# Features :
#   • Choice of C/C++ compiler (gcc / clang)
#   • Optional OpenMP support
#   • Automatic detection of GLFW (via pkg-config) and OpenGL
#   • Building of GLAD and libmeshb7 provided in the repository
#   • Generation of a Makefile via CMake → use 'make' or 'cmake --build <dir>' to compile
#
# CMake Options (pass at configure time):
#   -DUSE_OPENMP=ON/OFF    Enable or disable OpenMP (default: ON)
#   -DCMAKE_BUILD_TYPE=…   Specify build type (Debug, Release, RelWithDebInfo, etc.)
#
# Typical Usage:
#   Linux/macOS (pkg-config):
#     # Method from the root directory     
#     cmake -S . -B build -DUSE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
#     cmake --build build                  # portable: Make, Ninja, VS, Xcode...
#
#     # Method from the build directory
#     mkdir -p build && cd build
#     cmake .. -DUSE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
#     make  # or cmake --build .
#
#   Windows (vcpkg toolchain):
#     mkdir build && cd build
#     cmake -DUSE_OPENMP=ON \
#           -DCMAKE_BUILD_TYPE=Release \
#           -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
#           -B build -S .
#     cmake --build build                  # invokes msbuild, ninja, etc.
#     or cmake --build . --config Release  # Visual Studio users can specify config 
#
# Useful Variables (set via cmake-gui or -D):
#   CMAKE_C_COMPILER       Path to gcc/clang
#   CMAKE_CXX_COMPILER     Path to g++/clang++
#   CMAKE_CXX_STANDARD     C++ standard version (default: 17)
#   CMAKE_INSTALL_PREFIX   Installation directory (used by ‘make install’)

# ==============================================================================



# 1) Minimum CMake version required
cmake_minimum_required(VERSION 3.10)
#    -> CMake needs to be at least 3.10 to support modern functionalities 
#      (FetchContent, config options…).

# 2) Project declaration and language setup.
project(salami-opengl LANGUAGES C CXX)
#    -> Defines a C/C++ project named "salami-opengl".

# 3) Specifying the C++ standard
set(CMAKE_CXX_STANDARD 17) 
#    -> Equivalent to -std=c++17
set(CMAKE_CXX_EXTENSIONS OFF)
#    -> Disable compiler-specific extensions for portability  
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4) Options to enable/disable OpenMP
option(USE_OPENMP "Enable OpenMP support" ON)
#    -> Default is ON, to disable : 
#      cmake -DUSE_OPENMP=OFF ..

# 5) Prepare list of extra libraries (OpenMP, Threads)
set(EXTRA_LIBS "")
#   -> Initialize the EXTRA_LIBS variable with an empty list
if(USE_OPENMP)
  # -> If the USE_OPENMP option is on ...
  find_package(OpenMP)
  # -> Look for an OpenMP installation
  if(OpenMP_CXX_FOUND)
    # -> If OpenMP found for the C++	  
    message(STATUS "OpenMP found : ${OpenMP_CXX_LIBRARIES}")
    # Print the CMake configuration at the OpenMP lib line
    list(APPEND EXTRA_LIBS OpenMP::OpenMP_CXX)
    # -> Add the OpenMP::OpenMP_CXX target to EXTRA_LIBS
  else()
    message(WARNING "OpenMP not found, build without OpenMP")
  endif()
endif()

# 6) Systematically add the Threads lib (std::thread, etc)
find_package(Threads REQUIRED)
#    -> Search the Threads lib (posix threads on Linux)
#      CMake will fail if it is not found
list(APPEND EXTRA_LIBS Threads::Threads)
#    -> Add Threads to the extra linking list


# 7) Configure GLFW interface target
if(WIN32)
    # Windows : vcpkg must be active via -DCMAKE_TOOLCHAIN_FILE
    find_package(glfw3 CONFIG REQUIRED)    # Finds the imported target glfw3::glfw
    add_library(glfw3 INTERFACE IMPORTED)
    set_property(TARGET glfw3 PROPERTY
	    INTERFACE_LINK_LIBRARIES glfw3::glfw)

# -------- GLFW (interfaçage cross-platform) --------
if(APPLE)
  # 8a) macOS – Homebrew fournit glfw3Config.cmake et un target IMPORTED
  find_package(glfw3 CONFIG REQUIRED)

  # Wrapper INTERFACE “glfw3” sur le target importé
  add_library(glfw3 INTERFACE IMPORTED)
  if(TARGET GLFW::GLFW)
    # Propager include + link de GLFW::GLFW
    set_property(TARGET glfw3 PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES
        $<TARGET_PROPERTY:GLFW::GLFW,INTERFACE_INCLUDE_DIRECTORIES>)
    set_property(TARGET glfw3 PROPERTY
      INTERFACE_LINK_LIBRARIES GLFW::GLFW)
  elseif(TARGET glfw)
    set_property(TARGET glfw3 PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES
        $<TARGET_PROPERTY:glfw,INTERFACE_INCLUDE_DIRECTORIES>)
    set_property(TARGET glfw3 PROPERTY
      INTERFACE_LINK_LIBRARIES glfw)
  else()
    message(FATAL_ERROR "Imported GLFW target not found on macOS")
  endif()

elseif(UNIX)
  # 8b) Linux – on continue d’utiliser pkg-config
  find_package(PkgConfig REQUIRED)
  pkg_search_module(GLFW3 REQUIRED glfw3)

  add_library(glfw3 INTERFACE)
  target_include_directories(glfw3 INTERFACE ${GLFW3_INCLUDE_DIRS})
  target_link_libraries(glfw3 INTERFACE
    ${GLFW3_LDFLAGS}
    ${GLFW3_LIBRARIES}
  )

else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()



# 9) Find OpenGL (built-in CMake module)
find_package(OpenGL REQUIRED)
#    -> Attempt to find OpenGL (library + headers) on both Linux and macOS  
message(STATUS "OpenGL include dir   : ${OPENGL_INCLUDE_DIR}")
message(STATUS "OpenGL libraries     : ${OPENGL_LIBRARIES}")

# 10) Interface target for OpenGL
add_library(opengl_lib INTERFACE)
target_include_directories(opengl_lib INTERFACE ${OPENGL_INCLUDE_DIR})
target_link_libraries(opengl_lib INTERFACE ${OPENGL_LIBRARIES})
#    -> Add the OpenGL inclusion directory
#      On Linux we obtain -lGL, on macOS CMake will send -framework OpenGL.


# 11) Create the GLAD library from the provided source file src/glad.c
add_library(glad STATIC src/glad.c)
#     ->  Defines a STATIC library target "glad" from src/glad.c.
#       CMake will compile src/glad.c once into an archive (e.g. libglad.a).
#       Any executable that links to this target will reuse that single
#       compiled archive, avoiding recompilation for each binary.

# Specify include paths for GLAD headers
target_include_directories(glad
  PUBLIC
    include      # for glad.h
    include/KHR  # for khrplatform.h
    include/glad # for any additional GLAD headers
)
#    -> PUBLIC: propagates these -I flags to any target that links against "glad".

# 12) Create the meshb7 library from the provided source file
add_library(meshb7 STATIC libmeshb7.c)
#    -> Defines a STATIC library target "meshb7" from libmeshb7.c.
#       CMake will compile libmeshb7.c once into an archive (e.g. libmeshb7.a).

target_include_directories(meshb7
  PUBLIC
    include
)
#    -> PUBLIC: propagates this -I include flags to any target that links against "meshb7"

# 13) Useful function to create the executables
function(add_salami_target NAME SRC)
  add_executable(${NAME} ${SRC})
  # -> Create the NAME executable from SRC
  target_link_libraries(${NAME}
    PRIVATE
      glad
      meshb7
      glfw3
      opengl_lib
      ${EXTRA_LIBS}
  )
endfunction()
#   -> Automatically links GLAD, meshb7, OpenGL, GLFW and OpenMP/Threads (if found).

# 14) Declaring the three targets
add_salami_target(filter            filtershader.cpp)
add_salami_target(salami_lightning Salami_lightning.cpp)
add_salami_target(salami_window    Salami_movie_glwindow.cpp)

