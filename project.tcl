# create new HLS project for SRCNN
open_project srcnn_hls

# set top function for synthesis
set_top srcnn

# add source files
add_files src/srcnn.h
add_files src/srcnn.cpp
add_files src/conv1.cpp

# add testbench files
set CFLAGS "-I./src"
add_files -tb -cflags $CFLAGS ./test/csim.cpp
add_files -tb -cflags $CFLAGS ./test/tb_srcnn.cpp
add_files -tb -cflags $CFLAGS ./test/tb_conv1.cpp
add_files -tb -cflags $CFLAGS ./test/tb_set14.cpp
add_files -tb -cflags $CFLAGS ./test/util.h
add_files -tb -cflags $CFLAGS ./test/util.cpp

# add test dataset
add_files -tb ./src/weights/
add_files -tb ./test/set5/
add_files -tb ./test/set14/

# create new solution
open_solution "solution1"

# Set Kria SOM as target part
set_part  {xck26-sfvc784-2LV-c}

# Set default clock period to 10 ns
create_clock -period 10
