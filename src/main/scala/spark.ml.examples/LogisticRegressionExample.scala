package spark.ml.examples

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.sql.SparkSession


object LogisticRegressionExample {

  def main(): Unit = {

    val spark = SparkSession.builder().getOrCreate()

    // Loading data
    val data = spark.read.format("libsvm").load("/Users/Naresh/Desktop/sample_libsvm_data.txt")

    // logistic regression Model
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    //Fit the Model
    val lrModel = lr.fit(data)

    println(s"Coefficients: ${lrModel.coefficients} \n Intercepts : ${lrModel.intercept} ")

    val mlr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")

    val mlrModel = mlr.fit(data)

    println(s"Coefficients : ${mlrModel.coefficientMatrix} \n Intercepts: ${mlrModel.interceptVector}")

    // Summary of the logistic regression model
    val trainingSummary = lrModel.summary

    //Objective History per iteration
    val objectiveHistory = trainingSummary.objectiveHistory

    //We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
    // binary classification problem.
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    //spark.stop()
  }
}
