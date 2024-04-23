# CMake generated Testfile for 
# Source directory: C:/Users/User/Desktop/svm_main-posit1/cppposit-main/tests
# Build directory: C:/Users/User/Desktop/svm_main-posit1/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_init "C:/Users/User/Desktop/svm_main-posit1/build/tests/test_init.exe" "--success")
set_tests_properties(test_init PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/User/Desktop/svm_main-posit1/cppposit-main/tests/CMakeLists.txt;26;add_test;C:/Users/User/Desktop/svm_main-posit1/cppposit-main/tests/CMakeLists.txt;0;")
add_test(test_subnormals "C:/Users/User/Desktop/svm_main-posit1/build/tests/test_subnormals.exe" "--success")
set_tests_properties(test_subnormals PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/User/Desktop/svm_main-posit1/cppposit-main/tests/CMakeLists.txt;31;add_test;C:/Users/User/Desktop/svm_main-posit1/cppposit-main/tests/CMakeLists.txt;0;")
add_test(test_coverage "C:/Users/User/Desktop/svm_main-posit1/build/tests/test_subnormals.exe" "--success")
set_tests_properties(test_coverage PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/User/Desktop/svm_main-posit1/cppposit-main/tests/CMakeLists.txt;36;add_test;C:/Users/User/Desktop/svm_main-posit1/cppposit-main/tests/CMakeLists.txt;0;")
subdirs("../_deps/catch2-build")
