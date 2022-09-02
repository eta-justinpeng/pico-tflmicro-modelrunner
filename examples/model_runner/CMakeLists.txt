
cmake_minimum_required(VERSION 3.12)

project(model_runner C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(model_runner "")

target_include_directories(model_runner
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
    model_runner
    PROPERTIES
    COMPILE_FLAGS -fno-rtti
    COMPILE_FLAGS -fno-exceptions
    COMPILE_FLAGS -fno-threadsafe-statics
    COMPILE_FLAGS -nostdlib
)

target_sources(model_runner
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/model.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
  ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rp2/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

target_link_libraries(
  model_runner
  pico-tflmicro
  hardware_pwm
  pico_stdlib
)

pico_add_extra_outputs(model_runner)
pico_enable_stdio_usb(model_runner 1)
pico_enable_stdio_uart(model_runner 0)

