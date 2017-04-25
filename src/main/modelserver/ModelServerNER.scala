package main.modelserver

import java.io.Serializable

import cc.factorie.app.nlp.ner.NerTag
import cc.factorie.app.nlp.{Document, DocumentAnnotator, Sentence, Token}

import scala.reflect.ClassTag

/**
  * Created by Udit on 4/24/17.
  */
class ModelServerNER[L <: NerTag](modelServer: ModelServer)(implicit m: ClassTag[L]) extends DocumentAnnotator with
  Serializable {

  val prereqAttrs = Seq(classOf[Sentence])
  val postAttrs = Seq(m.runtimeClass)

  def process(document: Document): Document = {
    if (document.tokenCount > 0) {
      val processDocument = modelServer.processDocument(document)
      processDocument
    } else {
      document
    }
  }

  def tokenAnnotationString(token: Token): String = token.attr[L].categoryValue

}
