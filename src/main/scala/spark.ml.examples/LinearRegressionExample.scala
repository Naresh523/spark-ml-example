package spark.ml.examples

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearRegressionExample {

  def main(): Unit = {

    // Creating Spark Session
    val spark = SparkSession.builder().appName("Linear Regression Example").getOrCreate()

    //Loading csv data
    val csvData = spark.read.option("header", "true").option("inferSchema", "true").csv("")

    //Dataframe for ML takes in this format ("label", "features")
    import org.apache.spark.ml.feature.VectorAssembler

    // Some how df.select(col("")) is erroring syntax
    val df = csvData.select(csvData("Price").as("label"), csvData("Avg Area Income"), csvData("Avg Area House Age"), csvData("Avg Area Number of Rooms"),
      csvData("Avg Area Number of Bedrooms"), csvData("Area Population"))

    //This returns vector assembler object
    val assembler = new VectorAssembler()
      .setInputCols(Array("Avg Area Income", "Avg Area House Age", "Avg Area Number of Rooms",
        "Avg Area Number of Bedrooms", "Area Population"))
      .setOutputCol("features")

    val output = assembler.transform(df).select("label", "features")

    println(s"Output: ${output.show()}")

    //Linear Regression Object
    val lr = new LinearRegression()

    //Fit the model
    val lrModel = lr.fit(output)

    val trainingSummary = lrModel.summary
    trainingSummary.residuals.show()
    println(s"TrainingSummary : ${trainingSummary.predictions.show()}")
    println(s"RMSE : ${trainingSummary.rootMeanSquaredError}")

  }

}
