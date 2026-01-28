############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project finalproject_hls
set_top srcnn
add_files src/conv1.cpp
add_files src/srcnn.cpp
add_files src/srcnn.h
add_files -tb test/csim.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb test/display_bin.m -cflags "-Wno-unknown-pragmas"
add_files -tb test/tb_conv1.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb test/tb_set14.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb test/tb_srcnn.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb test/util.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb test/util.h -cflags "-Wno-unknown-pragmas"
add_files -tb test/set14 -cflags "-Wno-unknown-pragmas"
add_files -tb test/set5 -cflags "-Wno-unknown-pragmas"
add_files -tb test/weights -cflags "-Wno-unknown-pragmas"
open_solution "solution3" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
config_cosim -tool xsim
config_export -format ip_catalog -output C:/Users/ZIYUW12/Desktop/finalproject -rtl verilog
source "./finalproject_hls/solution3/directives.tcl"
csynth_design
