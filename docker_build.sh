# Create image with all compilers installed
docker build -t nebullvm-all .

# Create an image for each compiler installed
docker build -t nebullvm-onnxruntime . --build-arg COMPILER="onnxruntime"
docker build -t nebullvm-openvino . --build-arg COMPILER="openvino"
docker build -t nebullvm-tvm . --build-arg COMPILER="tvm"
docker build -t nebullvm-tensorrt . --build-arg COMPILER="tensorrt"
