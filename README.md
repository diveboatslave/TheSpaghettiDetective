# TheSpaghettiDetective - OpenVINO
This repo is a fork of [The Spaghetti Detective](https://github.com/TheSpaghettiDetective/TheSpaghettiDetective) project. 

The aim of this fork is to replace the YOLOv2 model used in the original project with an OpenVINO one that has been optimised to run quickly on CPUs, and not require the use of GPUs.

The [OpenVINO Toolkit](https://software.intel.com/en-us/openvino-toolkit) (Open Visual Inference and Neural network Optimization) was developed by Intel, and is a free toolkit that provides tools to convert Deep Learning models from several frameworks over to optimized models that run very quickly on Intel CPUs.

# Installing
See the documentation at [The Spaghetti Detective](https://github.com/TheSpaghettiDetective/TheSpaghettiDetective) for detailed information regarding software and hardware requirements, and server configuration.


Clone this fork
```bash
> git clone https://github.com/diveboatslave/TheSpaghettiDetective.git
```

Checkout the OpenVINO branch
```bash
> cd TheSpaghettiDetective
> git checkout openvino-model
```
# Running TSD in Docker

Building TSD with OpenVINO
```bash
> docker-compose -f dc-openvino.yml build
```

Start TSD using OpenVINO model
```bash
> docker-compose -f dc-openvino.yml up -d
```

Stopping TSD using OpenVINO model
```bash
> docker-compose -f dc-openvino.yml down
```



