# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\User\Desktop\svm_main-posit1\cppposit-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\User\Desktop\svm_main-posit1\build

# Utility rule file for examples.

# Include any custom commands dependencies for this target.
include examples/CMakeFiles/examples.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/examples.dir/progress.make

examples: examples/CMakeFiles/examples.dir/build.make
.PHONY : examples

# Rule to build all files generated by this target.
examples/CMakeFiles/examples.dir/build: examples
.PHONY : examples/CMakeFiles/examples.dir/build

examples/CMakeFiles/examples.dir/clean:
	cd /d C:\Users\User\Desktop\svm_main-posit1\build\examples && $(CMAKE_COMMAND) -P CMakeFiles\examples.dir\cmake_clean.cmake
.PHONY : examples/CMakeFiles/examples.dir/clean

examples/CMakeFiles/examples.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\User\Desktop\svm_main-posit1\cppposit-main C:\Users\User\Desktop\svm_main-posit1\cppposit-main\examples C:\Users\User\Desktop\svm_main-posit1\build C:\Users\User\Desktop\svm_main-posit1\build\examples C:\Users\User\Desktop\svm_main-posit1\build\examples\CMakeFiles\examples.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : examples/CMakeFiles/examples.dir/depend

