# tf-model-port: 
## Serving a pre-trained tensorflow model using Factorie

We use a pre-trained Tensorflow model (from [Fast and Accurate Sequence Labeling with Iterated Dilated Convolutions](https://arxiv.org/pdf/1702.02098.pdf)) and using Factorie, we serve the pre-trained model on a JVM. 

The input data is loaded and fed using Factorie documents.

A demo has been used to demonstrate this pre-trained model serving.

The pre-trained model can be found [here](https://drive.google.com/file/d/0BwSW2f4WefKyRW4taDA4b1k3TlE/view?usp=sharing). Create a `models` folder in the root of the project and put the `models.pb` file in this folder.

Additional dependencies: these should be put in the lib folder 
  - [Factorie](https://github.com/factorie/factorie) with pre-trained models >= 2.11
  
### TODO:
  - Benchmarking
  - Adding support for different models
  - Maven support
