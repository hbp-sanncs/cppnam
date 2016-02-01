cmake_minimum_required(VERSION 2.6)
project(cppnam)

add_executable(cppnam main.cpp)

install(TARGETS cppnam RUNTIME DESTINATION bin)
