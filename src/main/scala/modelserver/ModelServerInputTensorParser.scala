package main.scala.modelserver

import java.io.IOException
import java.nio.file.{Files, Path}
import java.util

import org.tensorflow.Tensor

/**
  * Created by Udit on 4/24/17.
  */
class ModelServerInputTensorParser {

  /**
    * Create the shape tensor from the shape of the batch
    *
    * @param batchTokenShapes token shapes of the batch
    * @return a tensor constructed from the batchTokenShapes
    */
  def getShapeTensor(batchTokenShapes: util.LinkedList[util.LinkedList[java.lang.Double]])
  : Tensor = {
    var shape_file_line = batchTokenShapes.get(0)
    val shape_file_line_size = shape_file_line.size
    val cols = shape_file_line.size
    val rows = batchTokenShapes.size
    val shape_long_buffer = Array.ofDim[Long](rows, cols)
    var i = 0
    while (i < rows) {
      shape_file_line = batchTokenShapes.get(i)
      var j = 0
      while (j < shape_file_line_size) {
        shape_long_buffer(i)(j) = shape_file_line.get(j).longValue
        j += 1
      }
      i += 1
    }
    var tensor = Tensor.create(shape_long_buffer)
    tensor
  }

  /**
    * Get the token tensor from the batch token indices
    *
    * @param batchTokenIndices the token indices from the batch
    * @return a tensor constructed from the batchTokenIndices
    */
  def getTokenTensor(batchTokenIndices: util.LinkedList[util.LinkedList[java.lang.Double]]):
  Tensor = {
    var token_file_line = batchTokenIndices.get(0)
    val token_file_line_size = token_file_line.size
    val cols = token_file_line.size
    val rows = batchTokenIndices.size
    val token_long_buffer = Array.ofDim[Long](rows, cols)
    var i = 0
    while (i < rows) {
      token_file_line = batchTokenIndices.get(i)
      var j = 0
      while (j < token_file_line_size) {
        token_long_buffer(i)(j) = token_file_line.get(j).longValue
        j += 1
      }
      i += 1
    }
    var tensor = Tensor.create(token_long_buffer)
    tensor
  }

  /**
    * Read all the bytes from the binary file
    *
    * @param path the path of the file
    * @return array of bytes read from the file
    */
  def readAllBytesOrExit(path: Path): Array[Byte] = {
    try
      return Files.readAllBytes(path)
    catch {
      case e: IOException =>
        e.printStackTrace()
        System.err.println("Failed to read [" + path + "]: " + e.getMessage)
        System.exit(1)
    }
    null
  }
}
