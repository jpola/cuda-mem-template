# On my machine where I have cuda and hcc installed together somehow hipcc detects cuda  and throws 
# #error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
# therefore below code should do the trick


hipcc --std=c++11 -dc -O3 -D__HIP_PLATFORM_HCC__ -I/home/jpola/External/HIP/include -I. -c memorytraverser.hip.cpp -o memorytraverser.o
echo "Memorytraverser compiled ... "
hipcc --std=c++11 -dc -O3 -D__HIP_PLATFORM_HCC__ -I/home/jpola/External/HIP/include -I. -c rotate_image_custom.hip.cpp -o rotate_image_custom.o
echo "Image rotation compiled ... "
g++ -o main.o -c -O3 --std=c++11 -D__HIP_PLATFORM_HCC__ -I/home/jpola/External/HIP/include -I/opt/hcc/include main.cpp
echo "Main compiled ... "

echo "Linking"
hipcc -o hip-traverser main.o memorytraverser.o rotate_image_custom.o -L/opt/hcc/lib -lstdc++ -lpthread -lm -lX11 -D__HIP_PLATFORM_HCC__
echo "done"
