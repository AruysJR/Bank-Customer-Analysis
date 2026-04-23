## Bank Customer Analysis & Subscription Prediction

An exploratory data analysis and machine learning project on a bank marketing dataset to understand customer behaviour and predict term deposit subscription.

### Dataset
Bank marketing dataset containing customer demographics, financial details, and campaign-related features such as job, marital status, education, balance, and previous outcomes.

### Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn (SMOTE)

### Key Insights
- Categorical features (job, marital status, education, contact, previous outcome) show clear patterns with subscription  
- Duration, balance, and age are the most influential numerical features  
- Customers with successful previous campaign outcomes are significantly more likely to subscribe  

---

### Modelling

**Baseline (Imbalanced Data)**
- Logistic Regression & Random Forest trained on raw data  
- High accuracy but **poor recall for subscribed customers**

**After SMOTE + Scaling**
- Applied SMOTE to balance classes  
- StandardScaler used for numerical features  
- Engineered age into a binary feature (≤50, >50)  
- Hyperparameter tuning with RandomizedSearchCV (StratifiedKFold, 5 folds)

---

### 📈 Results
- SMOTE significantly improved **recall and F1-score** for the minority class  
- Random Forest performed best overall after tuning  
- Balanced data improved the model’s ability to identify potential subscribers
