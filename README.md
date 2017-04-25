# factorie-tf-model-serve: 
### Serving a pre-trained tensorflow model using Factorie

We use a pre-trained Tensorflow model (from [Fast and Accurate Sequence Labeling with Iterated Dilated Convolutions](https://arxiv.org/pdf/1702.02098.pdf)) and using Factorie, we serve the pre-trained model on a JVM. 

The input data is loaded and fed using Factorie documents and the pre-trained model can be used as part of a Factorie pipeline.

The pre-trained model is served using [Tensorflow bindings](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary) on the JVM : 
A demo has been used to demonstrate this pre-trained model serving.

The pre-trained model can be found [here](https://drive.google.com/file/d/0BwSW2f4WefKyRW4taDA4b1k3TlE/view?usp=sharing). Create a `models` folder in the root of the project and put the `models.pb` file in this folder.

### Usage : Maven

To compile type
```
$ mvn compile
```
You might need additional memory.

```
export MAVEN_OPTS="$MAVEN_OPTS -Xmx1g -XX:MaxPermSize=128m"
```
To run the Demo file : `ModelServerDemo` run 
To run one of these examples using maven type
```
$ mvn scala:run -DmainClass=main.scala.ModelServerDemo
```

### TODO:
  - Benchmarking
  - Adding support for different models
