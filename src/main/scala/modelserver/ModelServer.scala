package main.scala.modelserver

import java.nio.file.Paths
import java.util

import cc.factorie.app.nlp.{Document, Sentence}
import org.tensorflow.{Graph, Session, Tensor}

/**
  * Created by Udit on 4/24/17.
  */
class ModelServer(inputTensorParser: ModelServerInputTensorParser,
                  dataPreprocessor: ModelServerDataPreprocessor,
                  modelPath: String) {

  val session = loadGraph(modelPath)

  /**
    * Run each document through the loaded graph
    *
    *
    */
  def processDocument(document: Document): Document = {

    /*dataPreprocessor.loadVocabulary()
    dataPreprocessor.loadShapeMap()
    dataPreprocessor.loadLabelMap()*/

    val sentences = document.sentences

    // load the frozen graph into a session


    // grouping the sentences into a batch of the denoted size
    val grouped = sentences.toArray.grouped(dataPreprocessor.batchSize)

    for (group <- grouped) {
      serveBatches(session, group)
    }

    // close the session
    session.close()
    document
  }

  /**
    * Load the frozen graph file (model.pb) into a session
    *
    * @param graphFile file of the graph
    * @return session in which the frozen graph has been loaded
    */
  def loadGraph(graphFile: String): Session = {
    println("Loading the model ...")
    val graph = new Graph
    val graphDef = inputTensorParser.readAllBytesOrExit(
      Paths.get(graphFile))

    // import the graph from the file, loading the
    // structure and weights of the graph
    if (graphDef != null) {
      graph.importGraphDef(graphDef)
    } else {
      throw new Exception("Had troubles reading from file. Graph is null.")
    }
    // start the session
    val session = new Session(graph)
    session
  }

  /**
    * Run each batch through the loaded graph
    *
    * @param session   the loaded session
    * @param sentences the batch of sentences to process
    */
  def serveBatches(session: Session, sentences: Iterable[Sentence]): Unit = {
    // find out the length of the longest sentence, to be used for padding
    var maxLength = 0
    for (sentence <- sentences) {
      if (maxLength <= sentence.length) {
        maxLength = sentence.length
      }
    }

    var batchTokenIndices = dataPreprocessor.mapTokensToIndices(sentences,
      maxLength)
    var batchTokenShapes = dataPreprocessor.mapTokensToShapes(sentences, maxLength)
    var seqLenMap = dataPreprocessor.getBatchSeqLength(sentences)

    // this tensor is mapped to the input tensor : input_x1
    val tokenTensor = inputTensorParser.getTokenTensor(batchTokenIndices)

    // this tensor is mapped to the input tensor : input_x2
    val shapeTensor_X2 = inputTensorParser.getShapeTensor(batchTokenShapes)

    // this tensor maps to the input tensor: batch_size
    val batch_size_tensor = Tensor.create(sentences.size)

    // this tensor maps to the input tensor: max_seq_len
    val max_seq_tensor = Tensor.create(maxLength + 2) // adding a padding for the beginning and
    // the end

    val predictedLabels = feedBatchToSession(session, shapeTensor_X2, tokenTensor,
      batch_size_tensor, max_seq_tensor)
    // map the predicted labels to NER tags, using the labelMap
    val labelsToTags = mapLabelsToTags(predictedLabels, seqLenMap)

    dataPreprocessor.populateTagsInSentences(sentences, labelsToTags)
  }

  /**
    * Feed the input tensors to the session, and fetch the output tensor
    *
    * @param session           the loaded session
    * @param shape             the shape tensor
    * @param token_tensor      the token tensor
    * @param batch_size_tensor the batch size tensor
    * @param max_seq_tensor    the max-seq-len tensor
    * @return the output tensor after feeding the batch through the graph
    */
  def feedBatchToSession(session: Session, shape: Tensor, token_tensor: Tensor, batch_size_tensor: Tensor,
                         max_seq_tensor: Tensor): Tensor = {
    val output = session.runner
      .feed("input_x1", token_tensor)
      .feed("input_x2", shape)
      .feed("max_seq_len", max_seq_tensor)
      .feed("batch_size", batch_size_tensor)
      .fetch("predictions/ArgMax").run.get(0)
    output
  }

  def getAccuracy(labels_map: util.HashMap[Integer, util.ArrayList[Long]], seq_len_map: util
  .HashMap[Integer, Integer], output: Tensor): Unit = {
    var outputShape = output.shape()
    var rows = java.lang.Integer.valueOf(String.valueOf(outputShape(0)))
    var cols = java.lang.Integer.valueOf(String.valueOf(outputShape(1)))
    val m = Array.ofDim[Long](rows, cols)
    val predictedLabels = new util.HashMap[Integer, util.ArrayList[Long]]
    val matrix = output.copyTo(m)
    var k = 0
    while (k < rows) {
      val row = new util.ArrayList[Long]
      var y = 0
      while (y < cols) {
        row.add(matrix(k)(y))
        y += 1
      }
      predictedLabels.put(k, row)
      k += 1
    }
    compare(labels_map, predictedLabels, seq_len_map)
  }

  def compare(labels: util.HashMap[Integer, util.ArrayList[Long]], predictedLabels: util.HashMap[Integer,
    util.ArrayList[Long]], seq_len_map: util.HashMap[Integer, Integer]): Unit = {
    var count = 0
    var total = 0
    var i = 0
    while (i < labels.size) {
      val row_label = labels.get(i)
      val row_predicted_labels = predictedLabels.get(i)
      val row_seq_len = seq_len_map.get(i)
      var t = 1
      while (t <= row_seq_len) {
        if (row_label.get(t).longValue == row_predicted_labels.get(t).longValue) count += 1
        total += 1
        t += 1
      }
      i += 1
    }
  }

  /**
    * Used to map the predicted/output labels to tags
    *
    * @param output    the predicted output
    * @param seqLenMap the sequence lengths of each of the sentences
    * @return the mapped sentences
    */
  def mapLabelsToTags(output: Tensor, seqLenMap: util.HashMap[Integer, Integer]):
  util.HashMap[Integer, util.ArrayList[String]] = {
    var outputShape = output.shape()
    var rows = java.lang.Integer.valueOf(String.valueOf(outputShape(0)))
    var cols = java.lang.Integer.valueOf(String.valueOf(outputShape(1)))
    val m = Array.ofDim[Long](rows, cols)
    val matrix = output.copyTo(m)
    val predictedLabels = new util.HashMap[Integer, util.ArrayList[String]]
    var k = 0
    while (k < rows) {
      val row = new util.ArrayList[String]
      val rowSeqLen = seqLenMap.get(k)
      var y = 1
      while (y < cols) {
        if (y <= rowSeqLen) {
          val token_label = dataPreprocessor.labelMap.get(String.valueOf(matrix(k)(y)).toInt)
          row.add(token_label)
        }
        y += 1
      }
      predictedLabels.put(k, row)
      k += 1
    }
    predictedLabels
  }
}
