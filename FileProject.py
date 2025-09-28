# =========================
# Connected with Google Drive
# =========================
USE_DRIVE = True  #When You want to read file from drive make True

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    INPUT_PATH = "/content/drive/BigDataTwitter/data2.csv"   # عدّل المسار حسب مكان الملف
    OUTPUT_DIR = "/content/drive/BigDataTwitter/data2_outputs"
else:
    INPUT_PATH = "/content/Sentiment140.csv"
    OUTPUT_DIR = "/content/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2) إنشاء SparkSession
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
# 3) تحميل Sentiment140
#    الأعمدة الأصلية: target,id,date,flag,user,text
#    القيم: 0 = سلبي ، 4 = إيجابي
# =========================
# ملاحظات:
# - بعض نسخ Sentiment140 تكون مفصولة بفواصل دون header.
# - إذا الملف مضغوط (.zip/.gz) فضّه أولاً أو عدّل القراءة.
df = (spark.read
      .option("header", False)
      .option("multiLine", True)
      .csv(INPUT_PATH)
      .toDF("target", "id", "date", "flag", "user", "text"))

# تنظيف أساسي
df = df.dropna(subset=["text"])
df = df.filter(F.length(F.col("text")) > 0)

# تحويل الهدف إلى 0/1
df = df.withColumn("label", F.when(F.col("target") == "4", 1.0).otherwise(0.0))

print("[INFO] Total rows:", df.count())
df.show(5, truncate=80)

# =========================
# 4) تجهيز النصوص (Tokenize -> Stopwords -> TF -> TFIDF)
# =========================
tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="tokens",
    pattern=r"[^A-Za-zء-ي0-9_]+",  # إنجليزي + عربي + أرقام
    gaps=True,
    toLowercase=True
)

# الإنجليزية كافتراضي لأن Sentiment140 إنجليزي غالبًا
stop_remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="tokens_clean",
    stopWords=StopWordsRemover.loadDefaultStopWords("english")
)

# خصّص حجم المفردات والدعم الأدنى لتسريع Colab إن لزم
cv = CountVectorizer(inputCol="tokens_clean", outputCol="tf", vocabSize=50000, minDF=5)
idf = IDF(inputCol="tf", outputCol="tfidf")

# =========================
# 5) تقسيم البيانات
# =========================
train, test = df.randomSplit([0.8, 0.2], seed=42)

# =========================
# 6) Logistic Regression
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
# 7) Random Forest
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
# 8) التقييم: Accuracy / Precision / Recall / F1
#    نستخدم المقاييس الموزونة (weighted) للاحتمال وجود عدم توازن
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

# إضافة أزمنة التدريب/التنبؤ
metrics_lr.update({"Model": "Logistic Regression",
                   "Training Time (sec)": lr_train_time,
                   "Prediction Time (sec)": lr_pred_time})
metrics_rf.update({"Model": "Random Forest",
                   "Training Time (sec)": rf_train_time,
                   "Prediction Time (sec)": rf_pred_time})

# ترتيب الأعمدة وحفظ النتائج
results_df = pd.DataFrame([metrics_lr, metrics_rf])[
    ["Model","Accuracy","Precision","Recall","F1","Training Time (sec)","Prediction Time (sec)"]
]
results_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
results_df.to_csv(results_path, index=False)
print("\n=== Evaluation Results ===")
display(results_df)
print(f"[INFO] Saved: {results_path}")

# =========================
# 9) Explainability (مبسّط)
#    - LR: معاملات كل كلمة (coefficient) -> أهم الكلمات إيجابيًا/سلبيًا
#    - RF: Feature Importances -> أهم الكلمات عالميًا
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
