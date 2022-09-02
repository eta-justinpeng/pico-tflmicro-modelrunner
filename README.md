# Pico TensorFlow Lite Micro Model Runner

## Setting up dependencies
Install the prerequisites for the Pico C/C++ SDK in the parent folder

```
cd ..
sudo apt update
sudo apt install git cmake gcc-arm-none-eabi gcc g++ libstdc++-arm-none-eabi-newlib
sudo apt install automake autoconf build-essential texinfo libtool libftdi-dev libusb-1.0-0-dev
```

In the parent folder, download the Pico C/C++ Repository and update the SDK

```
git clone -b master https://github.com/raspberrypi/pico-sdk.git
cd pico-sdk
git submodule update --init
```

Set the PICO_SDK_PATH environment variable to the installation directory of the pico-sdk

## Compiling the model
To run a model, first convert a tflite model to a .cpp file using xxd:

```
xxd -i your_model.tflite > model.cpp
```

Add the model.cpp file to pico-tflmicro-modelrunner/examples/model_runner

Create a cmake build folder in the pico-tflmicro-modelrunner directory

```
cd pico-tflmicro-modelrunner
mkdir build && cd build
cmake ..
```

To build the model_runner navigate to the examples folder:

```
cd examples
cd model_runner
```

Build the .uf2 binary

```
make
```

## Running the model on the pico

Plug in the pico while holding in the BOOTSEL button

Once the device appears, copy the model_runner.uf2 file to the device

Afterwards, the pico should unmount itself and restart.

To view output, use a serial terminal with a baud rate of 115200

## Some notes

The datatype for different models may need to be changed. To do this, change the int8_t datatypes in main_functions.cpp to whatever is required

Currently the model runner generates a random input. The random seed can be changed to further change the output. Or the input tensors could be added to a header file.