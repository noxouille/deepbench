# This script is for NVIDIA Driver 410.57

ERRMSG1="Please provide a device name (GPU/CPU)."

if   [ -z "$1" ]
then echo "No arguments supplied." $ERRMSG1;
     exit 1
fi

DOCKER_CMD="docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -v $(pwd):/home -w /home"
NGC_LINK="nvcr.io/nvidia"
CAFFE_TAG="18.08-py3" # This is the final version of Caffe2 container from NGC
TAG="18.09-py3"

DEVICE=$1
RESULTS_DIR="results/$DEVICE"

if   [ -d "${RESULTS_DIR}" ]
then echo "directory called ${RESULTS_DIR} already exist. Proceeding with benchmark...";
elif [ -f "${RESULTS_DIR}" ]
then echo "${RESULTS_DIR} is a file. Please consider moving/deleting it before running this script. Abort script."; exit 1;
else echo "Creating result subdirectory..."; mkdir -p $RESULTS_DIR;
fi

echo "Running Framework-wise benchmark with Caffe2:${CAFFE_TAG}..."
$DOCKER_CMD --shm-size=1g --ulimit memlock=-1 $NGC_LINK/caffe2:$CAFFE_TAG python3 framework_benchmark.py -f caffe2 > $RESULTS_DIR/$DEVICE-F-caffe2.txt
echo "Done!"

echo "Running Framework-wise benchmark with Pytorch:${TAG}..."
$DOCKER_CMD --ipc=host $NGC_LINK/pytorch:$TAG python3 framework_benchmark.py -f pytorch > $RESULTS_DIR/$DEVICE-F-pytorch.txt
echo "Done!"

echo "Running Framework-wise benchmark with TensorFlow:${TAG}..."
$DOCKER_CMD --shm-size=1g --ulimit memlock=-1 $NGC_LINK/tensorflow:$TAG python3 framework_benchmark.py -f tensorflow > $RESULTS_DIR/$DEVICE-F-tensorflow.txt
echo "Done!"

echo "Running Layer-wise benchmark with Pytorch:${TAG}..."
$DOCKER_CMD --ipc=host $NGC_LINK/pytorch:$TAG python3 layer_benchmark.py > $RESULTS_DIR/$DEVICE-L.txt
echo "Done!"

echo "Running Model-wise benchmark with Pytorch:${TAG}..."
$DOCKER_CMD --ipc=host $NGC_LINK/pytorch:$TAG python3 model_benchmark.py > $RESULTS_DIR/$DEVICE-M.txt
echo "Done!"
echo "All benchmarks done!"