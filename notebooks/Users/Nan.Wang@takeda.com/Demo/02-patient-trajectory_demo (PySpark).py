# Databricks notebook source
# MAGIC %md
# MAGIC ##<img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC 
# MAGIC ## Real World Evidence Data Analysis
# MAGIC ### Distributed ML with MLFlow and Hyperopt
# MAGIC In this notebook, we train a model to predict whether a patient is at risk of a given codition, using the patient's encounter history and demographic information using **pyspark**. 
# MAGIC 
# MAGIC <ol>
# MAGIC   <li> **Data**: We use the dataset in `rwe` database that we created using simulated patient records.</li>
# MAGIC   <li> **Parameteres**: Users can specify the target condition (to be predicted), the number of comorbid conditions to include, number of days of record, and training/test split fraction.
# MAGIC   <li> **Model Training**: We use [*spark ml*](https://spark.apache.org/docs/1.2.2/ml-guide.html)'s' random forest algorithm for binary classification. Moreover we use [*hyperopt*](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/index.html#hyperopt) for distributed hyperparameter tuning </li>
# MAGIC   <li> **Model tracking and management**: Using [*MLFlow*](https://docs.databricks.com/applications/mlflow/index.html#mlflow), we track our training experiments and log the models for each run </li>
# MAGIC </ol>

# COMMAND ----------

# DBTITLE 1,add text box for input parameters
dbutils.widgets.text('condition', '', 'Condition to model')
dbutils.widgets.text('num_conditions', '15', '# of comorbidities to include')
dbutils.widgets.text('days', '90', '# of days to use')
dbutils.widgets.text('training_set_size', '70', '% of samples for training set')

# COMMAND ----------

from pyspark.sql import Window
#import pyspark.sql.functions as F
from pyspark.sql.functions import *

# COMMAND ----------

import mlflow
import mlflow.spark 
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
import hyperopt
from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_OK

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Preparation and Exploration
# MAGIC To create the training data, we need to extract a dataset with both positive (affected ) and negative (not affected) labels.

# COMMAND ----------

# MAGIC %sql OPTIMIZE training_rwd.patient_encounters ZORDER BY (REASONDESCRIPTION, START_TIME, ZIP, PATIENT)

# COMMAND ----------

# MAGIC %sql
# MAGIC describe database training_rwd

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN training_rwd

# COMMAND ----------

# DBTITLE 1,load data for training
# load data from rwd database
patients = spark.read.table('training_rwd.patients')
encounters = spark.read.table('training_rwd.patient_encounters')

#.withColumn('REASONDESCRIPTION', F.when(col('REASONDESCRIPTION')=='Hyperlipidemia', 'Drug overdose').otherwise(col('REASONDESCRIPTION')))

# COMMAND ----------

print('column number:', len(encounters.columns), ' row number:', encounters.count())

# COMMAND ----------

display(encounters
  .where(col('REASONDESCRIPTION').isNotNull())
  .dropDuplicates(['PATIENT', 'REASONDESCRIPTION'])
  .groupBy('REASONDESCRIPTION').count()
  .orderBy('count', ascending=False)
  .limit(int(dbutils.widgets.get('num_conditions')))
)

# COMMAND ----------

# DBTITLE 1,Follow up Time Period
# MAGIC %sql
# MAGIC select distinct max(START_TIME) as LATEST_TIME, min(START_TIME) as EARLIEST_TIME
# MAGIC from training_rwd.patient_encounters

# COMMAND ----------

encounters.cube("encounterclass").count().orderBy("count", ascending=False).show()

# COMMAND ----------

encounters.createOrReplaceTempView('patient_encounters')

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct encounterclass, count(*) as count
# MAGIC from training_rwd.patient_encounters
# MAGIC group by encounterclass
# MAGIC order by count desc

# COMMAND ----------

pt_encounters = sql('select * from patient_encounters')

# COMMAND ----------

encounters.describe('cost').show()

# COMMAND ----------

# get the list of patients with the target condition (cases)
condition_patients = spark.sql("SELECT DISTINCT PATIENT FROM training_rwd.patient_encounters WHERE lower(REASONDESCRIPTION) LIKE '%" + dbutils.widgets.get('condition') + "%'")

# COMMAND ----------

# DBTITLE 1,List of patients with the condition to model
condition_patients = (encounters
  .where(lower(encounters.REASONDESCRIPTION).contains(dbutils.widgets.get('condition')))
  .select('PATIENT').dropDuplicates()
)

# COMMAND ----------

condition_patients.count()

# COMMAND ----------

display(encounters)

# COMMAND ----------

# DBTITLE 1,Number of patients for the top 10 co-morbid conditions
#create a dataframe of comorbid conditions
comorbid_conditions = (
 
  condition_patients.join(encounters, ['PATIENT'])
  .where(col('REASONDESCRIPTION').isNotNull())
  .dropDuplicates(['PATIENT', 'REASONDESCRIPTION'])
  .groupBy('REASONDESCRIPTION').count()
  .orderBy('count', ascending=False)
  .limit(int(dbutils.widgets.get('num_conditions')))
)

display(comorbid_conditions)

# COMMAND ----------

print('total number of patients: ', patients.count())
print('number of affected patients: ', condition_patients.count())

# COMMAND ----------



# COMMAND ----------

 .limit(condition_patients.count())

# COMMAND ----------

# DBTITLE 1,create list of positive and negative samples
# select balanced affected and unaffected patients

affected = condition_patients.count()
TotPT = patients.count()
if affected < TotPT/2:
  unaffected_patients = (
    patients
    .join(condition_patients,on=patients['Id'] == condition_patients['PATIENT'],how='left_anti')
    .sample(False, affected/(TotPT-affected))
    .select(col('Id').alias('PATIENT'))
  )
else:
  unaffected_patients = (
    patients.join(condition_patients,on=patients['Id'] == condition_patients['PATIENT'],how='left_anti')
    .select(col('Id').alias('PATIENT')))

  condition_patients = (
    condition_patients.sample(False, (TotPT-affected)/affected)
  )

# create a list of patients to include in training 
patients_to_study = condition_patients.union(unaffected_patients).cache()

# split dataste into train/test 
training_set_fraction = float(dbutils.widgets.get('training_set_size')) / 100.0

(train_patients, test_patients) = patients_to_study.randomSplit([training_set_fraction, 1.0 - training_set_fraction])

# COMMAND ----------

print('# of affected patients:', condition_patients.count(), '# of unaffected patients:', unaffected_patients.count())

# COMMAND ----------

print('# of training patients:', train_patients.count(), '# of test patients:', test_patients.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Engineering
# MAGIC Now that we have the data that we need, we create a function that takes the list of patient's to include, as well as an optional array of fitted indexers (which will be used for creating features in the test set) and outputs the dataset that will be used for calssification.

# COMMAND ----------

  lowest_date = (
    encounters
    .select(encounters['START_TIME'])
    .orderBy(encounters['START_TIME'])
    .limit(1)
    .withColumnRenamed('START_TIME', 'EARLIEST_TIME')
  )
  
  print('lowest_date: %s' % lowest_date.head(n=4))

# COMMAND ----------

display(lowest_date)

# COMMAND ----------

 encounter_features = (
     encounters.join(train_patients, on='PATIENT')
    .where(encounters['REASONDESCRIPTION'].isNotNull())
    .crossJoin(lowest_date)
    .withColumn("day", datediff(col('START_TIME'), col('EARLIEST_TIME')))
    .withColumn("patient_age", datediff(col('START_TIME'), col('BIRTHDATE')))
  )

# COMMAND ----------

display(encounter_features.where(col('REASONDESCRIPTION')=='Hyperlipidemia'))

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view timef as
# MAGIC select distinct PATIENT, max(START_TIME) as LATEST_TIME, min(START_TIME) as EARLIEST_TIME
# MAGIC from training_rwd.patient_encounters
# MAGIC group by PATIENT

# COMMAND ----------

lowest_date = spark.sql("select PATIENT, EARLIEST_TIME from timef")

# COMMAND ----------

lowest_date.count()

# COMMAND ----------

(encounters.join(train_patients, on='PATIENT')
  .where(encounters['REASONDESCRIPTION'].isNotNull())
 .count())

# COMMAND ----------

comorbidities = comorbid_conditions.collect()

# COMMAND ----------

comorbidities

# COMMAND ----------

  idx = 0
  for comorbidity in comorbidities:
    encounter_features = encounter_features.withColumn("comorbidity_%d" % idx,encounter_features['REASONDESCRIPTION'].like('%' + comorbidity['REASONDESCRIPTION'] + '%')).cache()
    idx += 1

# COMMAND ----------

display(encounter_features.where(col('PATIENT')=='53ce2b1f-2771-4041-9793-2c00c349de59'))

# COMMAND ----------

string_index_cols = []
strings_to_index = ['MARITAL', 'RACE', 'GENDER']
for string_to_index in strings_to_index:
      outCol = string_to_index + 'idx'
      string_index_cols.append(outCol) 
string_index_cols

# COMMAND ----------

string_index_cols = []
strings_to_index = ['MARITAL', 'RACE', 'GENDER']
string_indicers = []

for string_to_index in strings_to_index:
  outCol = string_to_index + 'idx'
  string_index_cols.append(outCol)

  si = StringIndexer(inputCol=string_to_index, outputCol=(outCol), handleInvalid='skip')
  model = si.fit(encounter_features)
  string_indicers.append(model)
  encounter_features = model.transform(encounter_features)
print(string_indicers, string_index_cols)
print(encounter_features)

# COMMAND ----------

display(encounter_features.where(col('PATIENT')=='53ce2b1f-2771-4041-9793-2c00c349de59')
        .select('recent_encounters', 'START_TIME', 'comorbidity_0', 'recent_0', 'comorbidity_1', 'recent_1', 'REASONDESCRIPTION'))

# COMMAND ----------

print(comorbidity_cols)
print(string_index_cols)

# COMMAND ----------

encounter_features.select('recent_encounters').withColumn('isNull_c',col('recent_encounters').isNull()).where('isNull_c = True').count()

# COMMAND ----------

encounter_features

# COMMAND ----------

# DBTITLE 1,define a function for feature engineering
def featurize_encounters(Pat, string_indicers=None):

  # get the first encounter date within the dataset
  lowest_date = (
    encounters
    .select(encounters['START_TIME'])
    .orderBy(encounters['START_TIME'])
    .limit(1)
    .withColumnRenamed('START_TIME', 'EARLIEST_TIME')
  )
  
  print('lowest_date: %s' % lowest_date.head(n=4))
  
  # get the last encounter date within the dataset
  encounter_features = (
     encounters.join(Pat, on='PATIENT')
    .where(encounters['REASONDESCRIPTION'].isNotNull())
    .crossJoin(lowest_date)
    .withColumn("day", datediff(col('START_TIME'), col('EARLIEST_TIME')))
    .withColumn("patient_age", datediff(col('START_TIME'), col('BIRTHDATE')))
  )
  
  # collect the list of comorbid conditions
  comorbidities = comorbid_conditions.collect()
  
  # now for each comorbid condition we add a feature column which indicates presense or absense of the condition for each patient in the training set
  idx = 0
  for comorbidity in comorbidities:
    encounter_features = encounter_features.withColumn("comorbidity_%d" % idx,encounter_features['REASONDESCRIPTION'].like('%' + comorbidity['REASONDESCRIPTION'] + '%')).cache()
    idx += 1
  
  string_index_cols = []
  strings_to_index = ['MARITAL', 'RACE', 'GENDER']
  
  # if the user specifies a list of string_indicers then those will be used (this is the case for when use it on the test data)
  if string_indicers:
    for model in string_indicers:
      encounter_features = model.transform(encounter_features)
    
    for string_to_index in strings_to_index:
      outCol = string_to_index + 'idx'
      string_index_cols.append(outCol)
      
  # creating an array of string indicers to transform categorical columns and adding transformed columns
  else:
    string_indicers = []
  
    for string_to_index in strings_to_index:
      outCol = string_to_index + 'idx'
      string_index_cols.append(outCol)
    
      si = StringIndexer(inputCol=string_to_index, outputCol=(outCol), handleInvalid='skip')
      model = si.fit(encounter_features)
      string_indicers.append(model)
      encounter_features = model.transform(encounter_features)
  
  # define a window function to include only records that are within the specified number of days 
  w = (
    Window.orderBy(encounter_features['day'])
    .partitionBy(encounter_features['PATIENT'])
    .rangeBetween(-int(dbutils.widgets.get('days')), -1)
  )
  # for each comorbidity add a column of the number of recent encounters
  comorbidity_cols = []
  for comorbidity_idx in range(idx):
    col_name = "recent_%d" % comorbidity_idx
    comorbidity_cols.append(col_name)
    
    encounter_features = encounter_features.withColumn(col_name, sum(col("comorbidity_%d" % comorbidity_idx).cast('int')).over(w)).\
      withColumn(col_name, expr("ifnull(%s, 0)" % col_name))
  
  # adding a column that indicates the number of recent encounters (within the specified number of days to use)
  encounter_features = encounter_features.withColumn("recent_encounters", count(lit(1)).over(w))
  
  # creating a vector of all features
  v = VectorAssembler(inputCols=comorbidity_cols + string_index_cols + ['patient_age', 'ZIP'], outputCol='features', handleInvalid='skip')
  encounter_features = v.transform(encounter_features)
  
  encounter_features = encounter_features.withColumn('label', encounter_features['comorbidity_0'].cast('int'))
  
  return (encounter_features, string_indicers)

# COMMAND ----------

(training_encounters, string_indicers) = featurize_encounters(train_patients)
display(training_encounters)

# COMMAND ----------

display(training_encounters)

# COMMAND ----------

display(training_encounters)

# COMMAND ----------

# MAGIC %md
# MAGIC ######class pyspark.ml.feature.VectorAssembler(inputCols=None, outputCol=None, handleInvalid='error')
# MAGIC A feature transformer that merges multiple columns into a vector column.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3. Model Training, Tracking and Hyperparameter tunning
# MAGIC 
# MAGIC Now train a binary classifier (using random forests) for predciting the target condition. We use MLFlow for tracking and registering the model, and use hyperopt for distributed hyper parameter tuning.

# COMMAND ----------

# DBTITLE 0,Temp fix for mlflow error message through load balancer
from databricks_cli.configure.provider import get_config_provider, set_config_provider, DatabricksConfigProvider
base_provider = get_config_provider()
  
class DynamicConfigProvider(DatabricksConfigProvider):
  def get_config(self):
    base_config = base_provider.get_config()
    base_config.insecure = True
    return base_config 
  
set_config_provider(DynamicConfigProvider())

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(training_encounters)

# COMMAND ----------

display(lrModel, training_encounters)

# COMMAND ----------

impurity = 'gini'
max_depth = 6
max_bins = 16

parameters = ['condition', 'num_conditions', 'days']

dt = DecisionTreeClassifier(impurity=impurity, maxDepth=max_depth, maxBins=max_bins)

model = dt.fit(training_encounters)

display(model, training_encounters)

(testing_encounters, _) = featurize_encounters(test_patients, string_indicers=string_indicers)

bce = BinaryClassificationEvaluator()
aroc = bce.evaluate(model.transform(testing_encounters))


# COMMAND ----------

display(model, testing_encounters)

# COMMAND ----------

mlflow.log_artifacts(image_dir, "images")

# COMMAND ----------

# DBTITLE 1,training function
def train(params):
  with mlflow.start_run():
    
    impurity = params['impurity']
    max_depth = int(params['max_depth'])
    max_bins = int(params['max_bins'])
    mlflow.log_param('impurity', impurity)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('max_bins', max_bins)
      
    parameters = ['condition', 'num_conditions', 'days']
    for parameter in parameters:
      mlflow.log_param(parameter, dbutils.widgets.get(parameter))
  
    dt = DecisionTreeClassifier(impurity=impurity, maxDepth=max_depth, maxBins=max_bins)
  
    model = dt.fit(training_encounters)
    mlflow.spark.log_model(model, 'patient-trajectory+PtAge')
  
    (testing_encounters, _) = featurize_encounters(test_patients, string_indicers=string_indicers)
  
    bce = BinaryClassificationEvaluator()
    test_transformed = model.transform(testing_encounters)
    aroc = bce.evaluate(test_transformed, {bce.metricName: "areaUnderROC"})
    aPR = bce.evaluate(test_transformed, {bce.metricName: "areaUnderPR"})
    
    # use sklearn to caluclate evaluation metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
    y_test = test_transformed.select('label').toPandas()
    y_pred = test_transformed.select('prediction').toPandas()
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test,y_pred)
    
    # get classification matrics as a dictionary
    class_report = classification_report(y_test,y_pred, output_dict=True)
    recall_0 = class_report['0']['recall']
    f1_score_0 = class_report['0']['f1-score']
    
    # log metrics
    mlflow.log_metric("accuracy_score", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall_0", recall_0)
    mlflow.log_metric("f1_score_0", f1_score_0)
    mlflow.log_metric("area_under_ROC", aroc)
    mlflow.log_metric("area_under_PR", aPR)
    
  return {'loss': -aroc, 'status': STATUS_OK}

# COMMAND ----------

    result = (test_transformed.cube('prediction', 'label').count()
            .where(col('prediction').isNotNull() & col('label').isNotNull())
            .withColumn('param', when((col('prediction')==1) & (col('label')==1), 'TP') 
                        .when((col('prediction')==0) & (col('label')==0), 'TN')
                        .when((col('prediction')==1) & (col('label')==0), 'FP')
                        .otherwise('FN'))
            .select('param', 'count').toPandas())
    
    # get confusion matrix values
    true_positive = result.iloc[0, 1]
    true_negative = result.iloc[3, 1]
    false_positive = result.iloc[1, 1]
    false_negative = result.iloc[2, 1]
    

# COMMAND ----------

# MAGIC %md
# MAGIC ######class pyspark.ml.classification.DecisionTreeClassifier(featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', seed=None)

# COMMAND ----------

# DBTITLE 1,Define the hyperparameter grid
criteria = ['gini', 'entropy']
search_space = {
  'max_depth': hyperopt.hp.uniform('max_depth', 2, 12),
  'max_bins': hyperopt.hp.choice('max_bins', [8, 16, 32]),
  'impurity': hyperopt.hp.choice('impurity', criteria)
}

# COMMAND ----------

import sys
sys.setrecursionlimit(100000)

# COMMAND ----------

# DBTITLE 1,Train the model and log the best model
# The algoritm to perform the parameter search
algo = tpe.suggest

argmin = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  max_evals=20)

# COMMAND ----------

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score

class RFCModel():

    def __init__(self, params={}):
        """
        Constructor for RandamForestClassifier
        :param params: parameters for the constructor such as no of estimators, depth of the tree, random_state etc
        """
        self._rf = RandomForestClassifier(**params)
        self._params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property
    def model(self):
        """
        Getter for the property the model
        :return: return the model
        """
        
        return self._rf
  
    @property 
    def params(self):
      """
      Getter for the property the model
        :return: return the model params
      """
      return self._params
    
    def mlflow_run(self, df, df_test, r_name="Lab-2:RF Bank Note Classification Experiment"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
          
            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id
            
            # train and predict
            self._rf.fit(df['features'], df['label'])
            X_test = df_test['features']
            y_test = df_test['label']
            y_pred = self._rf.predict(X_test)

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.model, "random-forest-class-model")
            mlflow.log_params(self.params)

            # compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test,y_pred)
            
            # ROC = summary of all confusion matrices that each
            # threshold produces
            roc = metrics.roc_auc_score(y_test, y_pred)

            # get confusion matrix values
            true_positive = conf_matrix[0][0]
            true_negative = conf_matrix[1][1]
            false_positive = conf_matrix[0][1]
            false_negative = conf_matrix[1][0]

            # get classification matrics as a dictionary
            class_report = classification_report(y_test,y_pred, output_dict=True)
            recall_0 = class_report['0']['recall']
            f1_score_0 = class_report['0']['f1-score']
            recall_1 = class_report['1']['recall']
            f1_score_1 = class_report['1']['f1-score']

            # log metrics
            mlflow.log_metric("accuracy_score", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("true_positive", true_positive)
            mlflow.log_metric("true_negative", true_negative)
            mlflow.log_metric("false_positive", false_positive)
            mlflow.log_metric("false_negative", false_negative)
            mlflow.log_metric("recall_0", recall_0)
            mlflow.log_metric("f1_score_0", f1_score_0)
            mlflow.log_metric("recall_1", recall_1)
            mlflow.log_metric("f1_score_1", f1_score_1)
            mlflow.log_metric("roc", roc)

            # create confusion matrix images
            (plt, fig, ax) = Utils.plot_confusion_matrix(y_test, y_pred, y, title="Bank Note Classification Confusion Matrix")

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("confusion_matrix-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "confusion_matrix_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimators trees:", self.params["n_estimators"])
            print(conf_matrix)
            print(classification_report(y_test,y_pred))
            print("Accuracy Score:", acc)
            print("Precision     :", precision)
            print("ROC           :", roc)

            return (experimentID, runID)


# COMMAND ----------

# DBTITLE 0,Random forest - sciki-learn
# iterate over several runs with different parameters
# TODO in the Lab (change these parameters, n_estimators and random_state
# with each iteration.
# Does that change the metrics and accuracy?
# start with n=10, step by 10 up to X <=40

(testing_encounters, _) = featurize_encounters(test_patients, string_indicers=string_indicers)

for n in range(10, 40, 10):
  params = {"n_estimators": n, "random_state": 42}
  rfr = RFCModel.new_instance(params)
  (experimentID, runID) = rfr.mlflow_run(df=training_encounters.select('features', 'label').toPandas(), 
                                         df_test=testing_encounters.select('features', 'label').toPandas())
  print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
  print("-" * 100)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## MLFlow dashboard
# MAGIC 
# MAGIC Now if you click on `Runs` in the top right corner of the notebook, you can see a list of runs of the notebook wich keeps track of parameters used in training, as well as performance metric (area under ROC in this case). For more information on using MLFlow dashboard and runs on databricks see [this blog](https://databricks.com/blog/2019/04/30/introducing-mlflow-run-sidebar-in-databricks-notebooks.html).