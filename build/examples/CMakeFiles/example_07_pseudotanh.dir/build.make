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

# Include any dependencies generated for this target.
include examples/CMakeFiles/example_07_pseudotanh.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/example_07_pseudotanh.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/example_07_pseudotanh.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/example_07_pseudotanh.dir/flags.make

examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj: examples/CMakeFiles/example_07_pseudotanh.dir/flags.make
examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj: examples/CMakeFiles/example_07_pseudotanh.dir/includes_CXX.rsp
examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj: C:/Users/User/Desktop/svm_main-posit1/cppposit-main/examples/07_pseudotanh/07_pseudotanh.cpp
examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj: examples/CMakeFiles/example_07_pseudotanh.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\User\Desktop\svm_main-posit1\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj"
	cd /d C:\Users\User\Desktop\svm_main-posit1\build\examples && C:\PROGRA~1\MinGW\ucrt64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj -MF CMakeFiles\example_07_pseudotanh.dir\07_pseudotanh\07_pseudotanh.cpp.obj.d -o CMakeFiles\example_07_pseudotanh.dir\07_pseudotanh\07_pseudotanh.cpp.obj -c C:\Users\User\Desktop\svm_main-posit1\cppposit-main\examples\07_pseudotanh\07_pseudotanh.cpp

examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.i"
	cd /d C:\Users\User\Desktop\svm_main-posit1\build\examples && C:\PROGRA~1\MinGW\ucrt64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\User\Desktop\svm_main-posit1\cppposit-main\examples\07_pseudotanh\07_pseudotanh.cpp > CMakeFiles\example_07_pseudotanh.dir\07_pseudotanh\07_pseudotanh.cpp.i

examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.s"
	cd /d C:\Users\User\Desktop\svm_main-posit1\build\examples && C:\PROGRA~1\MinGW\ucrt64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\User\Desktop\svm_main-posit1\cppposit-main\examples\07_pseudotanh\07_pseudotanh.cpp -o CMakeFiles\example_07_pseudotanh.dir\07_pseudotanh\07_pseudotanh.cpp.s

# Object files for target example_07_pseudotanh
example_07_pseudotanh_OBJECTS = \
"CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj"

# External object files for target example_07_pseudotanh
example_07_pseudotanh_EXTERNAL_OBJECTS =

examples/example_07_pseudotanh.exe: examples/CMakeFiles/example_07_pseudotanh.dir/07_pseudotanh/07_pseudotanh.cpp.obj
examples/example_07_pseudotanh.exe: examples/CMakeFiles/example_07_pseudotanh.dir/build.make
examples/example_07_pseudotanh.exe: examples/CMakeFiles/example_07_pseudotanh.dir/linkLibs.rsp
examples/example_07_pseudotanh.exe: examples/CMakeFiles/example_07_pseudotanh.dir/objects1.rsp
examples/example_07_pseudotanh.exe: examples/CMakeFiles/example_07_pseudotanh.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\User\Desktop\svm_main-posit1\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_07_pseudotanh.exe"
	cd /d C:\Users\User\Desktop\svm_main-posit1\build\examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\example_07_pseudotanh.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/example_07_pseudotanh.dir/build: examples/example_07_pseudotanh.exe
.PHONY : examples/CMakeFiles/example_07_pseudotanh.dir/build

examples/CMakeFiles/example_07_pseudotanh.dir/clean:
	cd /d C:\Users\User\Desktop\svm_main-posit1\build\examples && $(CMAKE_COMMAND) -P CMakeFiles\example_07_pseudotanh.dir\cmake_clean.cmake
.PHONY : examples/CMakeFiles/example_07_pseudotanh.dir/clean

examples/CMakeFiles/example_07_pseudotanh.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\User\Desktop\svm_main-posit1\cppposit-main C:\Users\User\Desktop\svm_main-posit1\cppposit-main\examples C:\Users\User\Desktop\svm_main-posit1\build C:\Users\User\Desktop\svm_main-posit1\build\examples C:\Users\User\Desktop\svm_main-posit1\build\examples\CMakeFiles\example_07_pseudotanh.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : examples/CMakeFiles/example_07_pseudotanh.dir/depend

