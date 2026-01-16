cmake_minimum_required(VERSION 3.10)
project(CompiledSolver)

# Set C and C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_library(FATROP_LIB
	NAMES fatrop
	PATHS /home/kwan/anaconda3/envs/wb-mpc/lib
	NO_DEFAULT_PATH
	REQUIRED
)
find_library(BLASFEO_LIB
	NAMES blasfeo
	PATHS /home/kwan/anaconda3/envs/wb-mpc/lib
	NO_DEFAULT_PATH
	REQUIRED
)

# Add include directories from the conda environment
include_directories(/home/kwan/anaconda3/envs/wb-mpc/include)

if(NOT FATROP_LIB)
	message(FATAL_ERROR "Could not find FATROP library. If installed in a custom location, set -DFATROP_LIB=/path/to/libfatrop.so")
endif()
if(NOT BLASFEO_LIB)
	message(FATAL_ERROR "Could not find BLASFEO library. If installed in a custom location, set -DBLASFEO_LIB=/path/to/libblasfeo.so")
endif()
# Add blasfeo include directory (adjust path as necessary)
include_directories(/opt/blasfeo/include)

# Add the source file
set(SOURCE_FILE solver_function.c)

# Create the shared library
add_library(solver_function SHARED ${SOURCE_FILE})

# Link the libraries
target_link_libraries(solver_function PRIVATE ${FATROP_LIB} ${BLASFEO_LIB})

# Compilation flags
target_compile_options(solver_function PRIVATE -O3 -fPIC)

# Install the shared library
install(TARGETS solver_function LIBRARY DESTINATION lib)
