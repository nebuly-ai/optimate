# Create image with all compilers installed
docker build -t nebulydocker/nebullvm:cuda11.2.0-nebullvm0.3.1-allcompilers .

# Create an image for each compiler installed
docker build -t nebulydocker/nebullvm:cuda11.2.0-nebullvm0.3.1-onnxruntime . --build-arg COMPILER="onnxruntime"
docker build -t nebulydocker/nebullvm:cuda11.2.0-nebullvm0.3.1-openvino . --build-arg COMPILER="openvino"
docker build -t nebulydocker/nebullvm:cuda11.2.0-nebullvm0.3.1-tvm . --build-arg COMPILER="tvm"
docker build -t nebulydocker/nebullvm:cuda11.2.0-nebullvm0.3.1-tensorrt . --build-arg COMPILER="tensorrt"
