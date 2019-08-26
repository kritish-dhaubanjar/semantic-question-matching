# Semantic Question Pair Matching with Deep Learning

### Sample Question Pair in Quora

![Quora Dataset](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/questions.png)

---

#### Distribution of Questions

![Distribution of Questions](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/count.png)

**System Architecture**
![System Architecture](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/arch.png)

### Stopwords Removal

![Stopwords Removal](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/stopwords.png)

### Stemming | Lemmatization

![Stemming](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/stem.png)

![Lemmatization](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/lemma.png)

### Word2vec

![Word2vec](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/word2vec.png)

### Correlation Heatmap for features

![Heatmap](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/heatmap.png)

### Distribution Plots

| len_q1 vs len_q2                                                                                                                 | diff_len                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| ![len_q1 vs len_q2](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/len.png) | ![diff_len](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/diff.png) |

### Scatter Plots

| cosine_distance vs common_words                                                                                       | fuzz_partial_ratio vs common_words                                                                                    |
| --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/scatter1.png) | ![](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/scatter2.png) |

## Supervised Machine Learning Models

| S.N. | Supervised Machine Learning | Accuracy |
| ---- | --------------------------- | -------- |
| 1    | Random Forest               | 0.7235   |
| 2    | K Nearest Neighbors         | 0.7104   |
| 3    | Logistic Regression         | 0.6680   |

## Neural Network

![Neural Network](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/assets/g719.png)

### Training Neural Network

![Training of Network](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/epoch.png)

| S.N. | Learning Rate | Batch Size | Training Accuracy |
| ---- | ------------- | ---------- | ----------------- |
| 1    | 0.01          | 30         | 0.6264            |
| 2    | 0.005         | 30         | 0.6938            |
| 3    | 0.001         | 30         | 0.7256            |

| Training Loss                                                                                                                  | Training Accuracy                                                                                                                      | ROC                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ![Training Loss](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/loss.png) | ![Training Accuracy](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/accuracy.png) | ![ROC](https://raw.githubusercontent.com/kritish-dhaubanjar/semantic-question-matching-latex/master/images/roc.png) |

### Confusion Matrix

|            |     | Predicted |       |
| ---------- | --- | --------- | ----- |
|            |     | 0         | 1     |
| **Actual** | 0   | 36462     | 12792 |
|            | 0   | 8954      | 20455 |

### Some Metrics

| Measure                   | Value  | Derivations                |
| ------------------------- | ------ | -------------------------- |
| Sensitivity               | 0.6955 | TPR = TP / (TP + FN)       |
| Specificity               | 0.7403 | SPC = TN / (FP + TN)       |
| Precision                 | 0.6152 | PPV = TP / (TP + FP)       |
| Negative Predictive Value | 0.8028 | NPV = TN / (TN + FN)       |
| False Positive Rate       | 0.2597 | FPR = FP / (FP + TN)       |
| False Discovery Rate      | 0.3848 | FDR = FP / (FP + TP)       |
| False Negative Rate       | 0.3045 | FNR = FN / (FN + TP)       |
| Accuracy                  | 0.7236 | ACC = (TP + TN) / (P + N)  |
| F1 Score                  | 0.6529 | F1 = 2TP / (2TP + FP + FN) |

> flask run --without-threads
