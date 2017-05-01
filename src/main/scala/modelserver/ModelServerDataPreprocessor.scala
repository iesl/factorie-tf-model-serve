package main.scala.modelserver

import java.util

import cc.factorie.app.nlp.Sentence
import cc.factorie.app.nlp.ner.BilouConllNerTag
import org.apache.commons.lang.StringUtils

import scala.io.Source

/**
  * Created by Udit on 4/24/17.
  */
trait DataPreprocessor {

  def loadVocabulary(): Unit

  def loadShapeMap(): Unit

  def loadLabelMap(): Unit

  def mapTokensToIndices(sentences: Iterable[Sentence],
                         maxLength: Int): util.LinkedList[util.LinkedList[java.lang.Double]]

  def mapTokensToShapes(sentences: Iterable[Sentence],
                        maxLength: Int): util.LinkedList[util.LinkedList[java.lang.Double]]

  def getBatchSeqLength(sentences: Iterable[Sentence]): util.HashMap[Integer, Integer]

  def populateTagsInSentences(sentences: Iterable[Sentence], labelsToTags:
  util.HashMap[Integer, util.LinkedList[String]]): Unit

}

object DataPreprocessor extends ModelServerDataPreprocessor(64,
  System.getProperty("user.dir") + "/config/vocabulary.txt",
  System.getProperty("user.dir") + "/config/shape.txt",
  System.getProperty("user.dir") + "/config/tags.txt")

class ModelServerDataPreprocessor(batchSz: Int, vocabularyFilePath: String, shapeFilePath: String,
                                  labelFilePath: String) extends DataPreprocessor {

  val batchSize = batchSz
  var vocabulary = new util.HashMap[String, Int]()
  var shapeMap = new util.HashMap[String, Int]()
  var labelMap = new util.HashMap[Integer, String]()

  /**
    * Load the vocabulary from a file into a hashmap
    *
    */
  def loadVocabulary(): Unit = {
    val count = 0
    for (line <- Source.fromFile(vocabularyFilePath).getLines) {
      val split = line.split("\t")
      if (split.size == 1) {
        vocabulary.put(" ", Integer.valueOf(split(0)))
      } else {
        vocabulary.put(split(0), Integer.valueOf(split(1)))
      }
    }
  }

  /**
    * Load the shapes from a file into a hash map
    *
    */
  def loadShapeMap(): Unit = {
    val count = 0

    for (line <- Source.fromFile(shapeFilePath).getLines) {
      val split = line.split("\t")
      shapeMap.put(split(0), Integer.valueOf(split(1)))
    }
  }

  /**
    * Load the labels into a hashmap from a file
    *
    */
  def loadLabelMap(): Unit = {
    val count = 0
    for (line <- Source.fromFile(labelFilePath).getLines) {
      val split = line.split("\t")
      labelMap.put(Integer.valueOf(split(1)), split(0))
    }
  }

  /**
    * Mapping of tokens to each index from the vocabulary
    *
    * @param sentences the batch of sentences
    * @param maxLength the length of the longest sentence in the batch
    * @return a batch of indexed sentences
    */
  def mapTokensToIndices(sentences: Iterable[Sentence],
                         maxLength: Int): util.LinkedList[util.LinkedList[java.lang.Double]] = {
    var batchSentenceList = new util.LinkedList[util.LinkedList[java.lang.Double]]()

    for (sentence <- sentences) {
      var batchSentence = processIndicesPerSentence(sentence, maxLength)
      batchSentenceList.add(batchSentence)
    }

    batchSentenceList
  }

  /**
    * Map each sentence's token to its index in the vocabulary
    *
    * @param sentence  the sentence who's token's index need to be mapped
    * @param maxLength the length of the longest sentence
    * @return the sentence, indexed
    */
  def processIndicesPerSentence(sentence: Sentence,
                                maxLength: Int): util.LinkedList[java.lang.Double] = {
    var batchSentence = new util.LinkedList[java.lang.Double]()
    // first/beginning padding
    batchSentence.add(0.0)

    // add the tokens' indices
    for (token <- sentence.tokens) {
      val index = vocabulary.get(token.string)
      batchSentence.add(new java.lang.Double(index))
    }

    // ending padding
    if (batchSentence.size() <= maxLength + 1) {
      while (batchSentence.size() <= maxLength + 1) {
        batchSentence.add(0.0)
      }
    }
    batchSentence
  }

  /**
    * Map tokens to shape indexes
    *
    * @param sentences the batch of sentences to map
    * @param maxLength the length of the longest sentence
    * @return the mapped sentence shapes
    */
  def mapTokensToShapes(sentences: Iterable[Sentence],
                        maxLength: Int): util.LinkedList[util.LinkedList[java.lang.Double]] = {
    var batchSentenceList = new util.LinkedList[util.LinkedList[java.lang.Double]]()

    for (sentence <- sentences) {
      var batchSentence = processShapesPerSentence(sentence, maxLength)
      batchSentenceList.add(batchSentence)
    }
    batchSentenceList
  }

  /**
    * Map each sentence's tokens to the their shape. (per sentence processing)
    *
    * @param sentence  the sentence whose tokens are to be mapped
    * @param maxLength the length of longest sentence
    * @return the mapped sentence
    */
  def processShapesPerSentence(sentence: Sentence,
                               maxLength: Int): util.LinkedList[java.lang.Double] = {
    var batchSentence = new util.LinkedList[java.lang.Double]()
    // first/beginning padding
    batchSentence.add(0.0)

    // mapping the tokens
    for (token <- sentence.tokens) {
      val shape = getShapeFromToken(token.string)
      val index = shapeMap.get(shape)
      batchSentence.add(new java.lang.Double(index))
    }

    // ending padding
    if (batchSentence.size() <= maxLength + 1) {
      while (batchSentence.size() <= maxLength + 1) {
        batchSentence.add(0.0)
      }
    }

    batchSentence
  }

  /**
    * Map each token to its shape:
    * < all upper case > - "AA"
    * < all lower case or digits > - "a"
    * < first letter capital, rest lower case > - "Aa"
    * < first letter lower case, some letter in the middle capitalized > - "aAa"
    *
    * @param string
    * @return
    */
  def getShapeFromToken(string: String): String = {
    var tag = "a"
    if (StringUtils.isAllUpperCase(string)) {
      tag = "AA";
    } else if (StringUtils.isAllLowerCase(string)) {
      tag = "a";
    } else if (string.charAt(0).isUpper) {
      tag = "Aa";
    } else {
      val sz = string.size
      var i = 0
      for (i <- 0 to sz - 1) {
        if (string.charAt(i).isUpper) {
          tag = "aAa"
        }
      }
    }
    tag
  }

  /**
    * Get the sequence length of each of the sentences in the current batch
    *
    * @param sentences the current batch of sentences
    * @return the sequence lengths of sentences
    */
  def getBatchSeqLength(sentences: Iterable[Sentence]): util.HashMap[Integer, Integer] = {
    val seqLenMap = new util.HashMap[Integer, Integer]()
    var count = 0
    for (sentence <- sentences) {
      seqLenMap.put(count, sentence.tokens.size)
      count += 1
    }
    seqLenMap
  }

  /**
    * Populate tags in sentences
    *
    * @param sentences
    * @param labelsToTags
    */
  def populateTagsInSentences(sentences: Iterable[Sentence], labelsToTags:
  util.HashMap[Integer, util.LinkedList[String]]): Unit = {
    val rows = labelsToTags.size()
    var i = 0
    for (sentence <- sentences) {
      populateTagPerSentence(sentence, labelsToTags.get(i))
      i += 1
    }
  }

  /**
    * Populate per sentence tags
    *
    * @param sentence
    * @param strings
    */
  def populateTagPerSentence(sentence: Sentence, strings: util.LinkedList[String]): Unit = {
    val size = strings.size()
    var i = 0
    for (token <- sentence.tokens) {
      token.attr += new BilouConllNerTag(token = token, strings.get(i))
      i += 1
    }
  }
}
