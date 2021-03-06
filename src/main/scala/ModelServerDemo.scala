package main.scala

import java.io.File

import cc.factorie.app.nlp.DocumentAnnotationPipeline
import cc.factorie.app.nlp.load.LoadPlainText
import cc.factorie.app.nlp.ner.BilouConllNerTag
import cc.factorie.app.nlp.segment.{DeterministicNormalizingTokenizer, DeterministicSentenceSegmenter}
import main.scala.modelserver._

import scala.io.Source

/**
  * Created by Udit on 4/18/17.
  */

object ModelServerDemo {
  def main(args: Array[String]): Unit = {
    val currentDir = System.getProperty("user.dir")
    val modelPath = currentDir + "/models/model.pb"

    object modelServer extends ModelServer(InputTensorParser, DataPreprocessor, modelPath)
    object modelServerNER extends ModelServerNER[BilouConllNerTag](modelServer)

    val documents = LoadPlainText.fromSource(Source.fromInputStream(getClass().getResourceAsStream("/demo/input_3.txt")))

    val annotators = Seq(DeterministicNormalizingTokenizer, DeterministicSentenceSegmenter, modelServerNER)
    val pipeline = new DocumentAnnotationPipeline(annotators)

    var start = System.nanoTime()
    for (doc <- documents) {
      pipeline.process(doc)

      println(s"sentences: ${doc.sentenceCount} tokens: ${doc.tokenCount}")

      /*doc.sentences.foreach { s =>
        s.tokens.foreach { t =>
          println(s"${t.positionInSentence}\t${t.string}\t${t.nerTag}")
        }
      }*/
    }
    var end = System.nanoTime()
    println("Time taken = " + ((end-start)/(1000000) + " ms"))
  }
}
