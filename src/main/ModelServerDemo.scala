package main

import java.io.File

import cc.factorie.app.nlp.DocumentAnnotationPipeline
import cc.factorie.app.nlp.load.LoadPlainText
import cc.factorie.app.nlp.ner.{BilouConllNerTag, ConllChainNer, StaticLexiconFeatures}
import cc.factorie.app.nlp.segment.{DeterministicNormalizingTokenizer, DeterministicSentenceSegmenter}
import cc.factorie.util.ModelProvider
import main.modelserver.{ModelServer, ModelServerDataPreprocessor, ModelServerInputTensorParser, ModelServerNER}

import scala.io.Source

/**
  * Created by Udit on 4/18/17.
  */

object ModelServerDemo {
  def main(args: Array[String]): Unit = {
    val currentDir = System.getProperty("user.dir")
    val inputDataFileName = currentDir + "/data/input_3.txt"
    val inputDataShapeFileName = currentDir + "/config/shape.txt"
    val tagMapFileName = currentDir + "/config/tags.txt"
    val vocabFileName = currentDir + "/config/vocabulary.txt"
    val modelPath = currentDir + "/models/model.pb"

    object dataPreprocessor extends ModelServerDataPreprocessor(100)
    dataPreprocessor.loadVocabulary(vocabFileName)
    dataPreprocessor.loadShapeMap(inputDataShapeFileName)
    dataPreprocessor.loadLabelMap(tagMapFileName)

//    var documents = LoadPlainText.fromSource(Source.fromFile(file = new File(inputDataFileName)))

    object inputTensorParser extends ModelServerInputTensorParser

    object modelServer extends ModelServer(inputTensorParser, dataPreprocessor, modelPath)
    object modelServerNER extends ModelServerNER[BilouConllNerTag](modelServer)

    val documents = LoadPlainText.fromSource(Source.fromFile(file = new File
    (inputDataFileName)))

    val annotators = Seq(DeterministicNormalizingTokenizer, DeterministicSentenceSegmenter, modelServerNER)
    val pipeline = new DocumentAnnotationPipeline(annotators)

    for (doc <- documents) {
      pipeline.process(doc)

      println(s"sentences: ${doc.sentenceCount} tokens: ${doc.tokenCount}")

      doc.sentences.foreach { s =>
        s.tokens.foreach { t =>
          println(s"${t.positionInSentence}\t${t.string}\t${t.nerTag}")
        }
      }
    }
  }
}
