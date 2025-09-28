# =========================
# Connected with Google Drive
# =========================
USE_DRIVE = True  #When You want to read file from drive make True

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    INPUT_PATH = "/content/drive/BigDataTwitter/data2.csv"   #Your path for dataset in drive
    OUTPUT_DIR = "/content/drive/BigDataTwitter/data2_outputs" #Your path output in drive
else:
    INPUT_PATH = "/content/data2.csv" #Your path for dataset in local
    OUTPUT_DIR = "/content/data2_outputs"#Your path output in local
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Create SparkSession
# =========================
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import pandas as pd
import numpy as np

spark = (SparkSession.builder
         .appName("TwitterSentiment_RF_LR_Explainable")
         .config("spark.ui.showConsoleProgress", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

print("[INFO] Spark version:", spark.version)

# =========================
# Load Dataset "Sentiment140"
#    Column: target,id,date,flag,user,text
# =========================
df = (spark.read
      .option("header", False)
      .option("multiLine", True)
      .csv(INPUT_PATH)
      .toDF("target", "id", "date", "flag", "user", "text"))

# PrePrcoessing data
# Delete row in text column have NULL or empty.
df = df.dropna(subset=["text"])
df = df.filter(F.length(F.col("text")) > 0)

#Exclude tweets Neutral 
df = df.filter(F.col("target").isin("0", "4"))

# Binarization target column to 0/1 (1.0 positvie , 0.0 negative)
df = df.withColumn("label", F.when(F.col("target") == "4", 1.0).otherwise(0.0))

print("[INFO] Total rows:", df.count())
df.show(5, truncate=80)

# =========================
# Cleant text (Tokenize -> Stopwords -> TF -> TFIDF)
# =========================
tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="tokens",
    pattern=r"[^A-Za-zء-ي0-9_]+",  # إنجليزي + عربي + أرقام
    gaps=True,
    toLowercase=True
)

# Stop removal Defualt English word.
stop_remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="tokens_clean",
    stopWords=StopWordsRemover.loadDefaultStopWords("english")
)

cv = CountVectorizer(inputCol="tokens_clean", outputCol="tf", vocabSize=50000, minDF=5)
idf = IDF(inputCol="tf", outputCol="tfidf")

# =========================
# Split data to train 70%, test 30%.    
# =========================
train, test = df.randomSplit([0.7, 0.3], seed=42)

# =========================
# Implement Logistic Regression Model
# =========================
lr = LogisticRegression(featuresCol="tfidf", labelCol="label", maxIter=50, regParam=0.0)
pipeline_lr = Pipeline(stages=[tokenizer, stop_remover, cv, idf, lr])

t0 = time.time()
lr_model = pipeline_lr.fit(train)
lr_train_time = time.time() - t0

t0 = time.time()
lr_pred = lr_model.transform(test).cache()
lr_pred_time = time.time() - t0

# =========================
# Implement Random Forest Model
# =========================
rf = RandomForestClassifier(featuresCol="tfidf", labelCol="label",
                            numTrees=100, maxDepth=10, seed=42)
pipeline_rf = Pipeline(stages=[tokenizer, stop_remover, cv, idf, rf])

t0 = time.time()
rf_model = pipeline_rf.fit(train)
rf_train_time = time.time() - t0

t0 = time.time()
rf_pred = rf_model.transform(test).cache()
rf_pred_time = time.time() - t0

# =========================
# Evalution metrics (Accuracy, Precision, Recall, F1)
# =========================
def evaluate_all(pred_df, label_col="label", pred_col="prediction"):
    eval_acc = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="accuracy")
    eval_prec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="weightedPrecision")
    eval_rec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="weightedRecall")
    eval_f1  = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="f1")
    return {
        "Accuracy": float(eval_acc.evaluate(pred_df)),
        "Precision": float(eval_prec.evaluate(pred_df)),
        "Recall": float(eval_rec.evaluate(pred_df)),
        "F1": float(eval_f1.evaluate(pred_df))
    }

metrics_lr = evaluate_all(lr_pred)
metrics_rf = evaluate_all(rf_pred)

# Training Time and Prediction Time For models 
metrics_lr.update({"Model": "Logistic Regression",
                   "Training Time (sec)": lr_train_time,
                   "Prediction Time (sec)": lr_pred_time})
metrics_rf.update({"Model": "Random Forest",
                   "Training Time (sec)": rf_train_time,
                   "Prediction Time (sec)": rf_pred_time})

# Save result
results_df = pd.DataFrame([metrics_lr, metrics_rf])[
    ["Model","Accuracy","Precision","Recall","F1","Training Time (sec)","Prediction Time (sec)"]
]
results_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
results_df.to_csv(results_path, index=False)
print("\n=== Evaluation Results ===")
display(results_df)
print(f"[INFO] Saved: {results_path}")

# =========================
# Explainability
#    - Logistic Regression: Top important negative/positive word
#    - Random Forest: Top Feature Importances
# =========================
def save_top_terms_from_lr(pipeline_model, top_k=30):
    # tokenizer(0) -> stop_remover(1) -> cv(2) -> idf(3) -> lr(4)
    cv_model = pipeline_model.stages[2]
    lr_model = pipeline_model.stages[4]
    vocab = cv_model.vocabulary
    coeff = lr_model.coefficients.toArray()  # نفس ترتيب vocab

    pos_idx = np.argsort(coeff)[-top_k:][::-1]
    neg_idx = np.argsort(coeff)[:top_k]

    top_pos = [(vocab[i], float(coeff[i])) for i in pos_idx]
    top_neg = [(vocab[i], float(coeff[i])) for i in neg_idx]

    pos_path = os.path.join(OUTPUT_DIR, "lr_top_positive_terms.csv")
    neg_path = os.path.join(OUTPUT_DIR, "lr_top_negative_terms.csv")
    pd.DataFrame(top_pos, columns=["token","coefficient"]).to_csv(pos_path, index=False)
    pd.DataFrame(top_neg, columns=["token","coefficient"]).to_csv(neg_path, index=False)
    print(f"[Explain-LR] Saved:\n  {pos_path}\n  {neg_path}")

def save_top_terms_from_rf(pipeline_model, top_k=50):
    # tokenizer(0) -> stop_remover(1) -> cv(2) -> idf(3) -> rf(4)
    cv_model = pipeline_model.stages[2]
    rf_model = pipeline_model.stages[4]
    vocab = cv_model.vocabulary
    importances = np.array(rf_model.featureImportances.toArray())  # طولها = حجم الميزات
    idx = np.argsort(importances)[-top_k:][::-1]
    top_feats = [(vocab[i], float(importances[i])) for i in idx]
    path = os.path.join(OUTPUT_DIR, "rf_top_features.csv")
    pd.DataFrame(top_feats, columns=["token","importance"]).to_csv(path, index=False)
    print(f"[Explain-RF] Saved:\n  {path}")

save_top_terms_from_lr(lr_model, top_k=30)
save_top_terms_from_rf(rf_model, top_k=50)

# =========================
# 10) تفسير على مستوى التغريدة (LR)
#     مساهمة كل توكن تقريبًا = TFIDF(token) * LR_coefficient(token)
#     نعيد أهم 10 توكنات لكل تغريدة
# =========================
# استرجاع vocab و coeff مرة أخرى من نموذج LR
cv_model_lr = lr_model.stages[2]
lr_stage = lr_model.stages[4]
VOCAB = cv_model_lr.vocabulary
COEFF = lr_stage.coefficients.toArray()

@F.udf(returnType=T.ArrayType(T.StructType([
    T.StructField("token", T.StringType()),
    T.StructField("contribution", T.DoubleType())
])))
def explain_vector(tfidf_sparse):
    if tfidf_sparse is None:
        return []
    # tfidf_sparse: SparseVector with .indices and .values
    contribs = []
    try:
        idxs = tfidf_sparse.indices
        vals = tfidf_sparse.values
    except Exception:
        return []
    for idx, val in zip(idxs, vals):
        if idx < len(VOCAB):
            token = VOCAB[idx]
        else:
            token = f"idx_{idx}"
        c = float(val) * float(COEFF[idx])
        contribs.append({"token": token, "contribution": c})
    # رتّب حسب القيمة المطلقة للمساهمة
    contribs = sorted(contribs, key=lambda d: abs(d["contribution"]), reverse=True)[:10]
    return contribs

# أعد حساب ميزات LR على test للحصول على tfidf (نفس pipeline)
lr_features_test = lr_model.transform(test).select("text","label","prediction","tfidf")

explanations = (lr_features_test
                .withColumn("explanations", explain_vector(F.col("tfidf")))
                .withColumn("top_pos_tokens",
                            F.expr("slice(transform(filter(explanations, x -> x.contribution > 0), x -> x.token), 1, 5)"))
                .withColumn("top_neg_tokens",
                            F.expr("slice(transform(filter(explanations, x -> x.contribution < 0), x -> x.token), 1, 5)"))
                .select("text","label","prediction","top_pos_tokens","top_neg_tokens","explanations"))

explain_path = os.path.join(OUTPUT_DIR, "tweet_level_explanations.json")
(explanations
 .limit(2000)  # لمنع ملفات ضخمة في كولاب، غيّر الرقم حسب الحاجة
 .coalesce(1)
 .write.mode("overwrite")
 .json(explain_path))
print(f"[Explain-PerTweet] Saved (sample) to: {explain_path}")

# عرض عيّنة صغيرة
print("\n=== Sample tweet-level explanations (LR) ===")
for row in explanations.limit(5).collect():
    print("Text:", row["text"][:120].replace("\n"," "))
    print("Pred:", int(row["prediction"]), "| Top+:", row["top_pos_tokens"], "| Top-:", row["top_neg_tokens"])
    print("----")

print("\n[DONE] كل النتائج تم حفظها في:", OUTPUT_DIR)
