# House Prices - Advanced Regression Techniques
---
# კონკურსის მოკლე აღწერა
House Prices კონკურსის მიზანაია სახლების მოცემული მახასიათებლების საფუძველზე მათი ფასების პროგნოზირება. 


# სტრუქტურა

```
House_Prices/
│
├── house-model-experiment.ipynb     ← cleaning, feature engineering, feature selection, training. 
├── house-model-inference.ipynb      ← prediction-submission.
├── README.md                        ← Detailed description for each approach.
```

---

# ფაილების აღწერა
| notebook | description |
|---|---|
| `house-model-experiment.ipynb`| მთავარი notebook მოდელების ექსპერიმენტებისათვის / EDA |
| `house-model-inference.ipynb` | MLflow-იდან pipeline-ისა და მოდელის ჩატვირთვა, Kaggle-ის ტესტ სეტის ტრანსფორმაცია და submission გენერაცია |

---

## მონაცემთა გაწმენდა და დამუშავება (Cleaning & Preprocessing)

**I. სვეტების ანალიზი NA/ცარიელი მნიშვნელობებით:**


<img width="1500" height="700" alt="image" src="https://github.com/user-attachments/assets/afd533c0-e8cc-454f-adb0-5799bdd8f1fe" />


**სვეტები მაღალი NA პროპორციებით:**
  *  PoolQC ~ 100%
  *  MiscFeature ~ 95%
  *  Alley ~ 95%
  *  Fence ~ 80%

მოცემულ სვეტებს (>80% NA) დავდროპავ, მეორემხრივ თუ ამ ცარიელ მნიშვნელობებს შევავსებ, დიდი რისკია overfitting-ში გადასვლის.

**II. ნულოვანი/ერთგვაროვანი სვეტების მოშორება**
 * numerical სვეტები სადაც ~95% მნიშვნელობა იყო 0 :
   - `LowQualFinSF`, `3SsnPorch`, `PoolArea`, `MiscVal`
 * categorical სვეტები სადაც ~95% მნიშვნელობა ერთი კატეგორია იყო:
   - `Street`, `Utilities`, `Condition2`, `RoofMatl`, `Heating`, `GarageQual`, `GarageCond`
   
(ლექტორის რეკომენდაციის შემდეგ, რომ ბევრი პლოტი აგვეგო, თითოეული feature-ის distribution plot-ით ვნახე, რომ ბევრ სვეტში ერთიდაიგივე მნიშვნელობები გვხვდებოდა. შესაბამისად თუ feature-ის მხოლოდ ~1 მნიშვნელობა არსებობს ვყრი მსგავს სვეტს. notebook-ში ყველა სვეტის distributiin plot-ია, შეგიძლიათ ნახოთ.) 

**მაგალითისთვის :**
---
<img width="291" height="270" alt="image" src="https://github.com/user-attachments/assets/9328963f-da96-4853-b891-43c871f897fa" />
<img width="272" height="268" alt="image" src="https://github.com/user-attachments/assets/fe6d3e4d-0b8c-4903-92ce-82698911dc5a" />
<img width="274" height="271" alt="image" src="https://github.com/user-attachments/assets/b9474df7-f2a3-4edc-b62c-40b558d253a9" />



III. NA მნიშვნელობების შევსება
 * **numerical columns** - მედიანა
 * **categorial columns** - მოდა

##  Feature Engineering

* **ახალი სვეტები**

| Feature | Join_Formula | Info |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | საერთო ფართობი |
| `TotalBath` | `FullBath + BsmtFullBath + 0.5*HalfBath + 0.5*BsmtHalfBath` | სველი წერტილების საერთო რაოდენობა |
| `HouseAge` | `YrSold - YearBuilt` | სახლის ასაკი გაყიდვის მომენტში |
| `Remodeled` | `YearRemodAdd != YearBuilt` | განახლებულია თუ არა |

* **Encoding**
**categorical -> numerical** სვეტები დავყავი unique მნიშვნელობების რაოდენობის მიხედვით (threshold = 3):

  * high cardinality (>3 unique) : Target Encoding, კატეგორია იცვლება საშუალო SalePrice-ით
  * low cardinality (≤3 unique)  : One Hot Encoding, მარტივი კატეგორიებისთვის 

## Feature Selection

**I. Correlation Filter (threshold=0.85):**
ერთმანეთთან მაღალ კორელაციაში მყოფი სვეტებიდან უფრო redundant-ი წაიშალა. შედეგად 66 სვეტი დამრჩა.

**II. RFE - Recursive Feature Elimination (n=20):**
LinearRegression-ზე დაფუძნებული RFE-ით შეირჩა 20 ყველაზე მნიშვნელოვანი სვეტი


## ტრენინგი და ექსპერიმენტები

გამოვიყენე ოთხი ძირითადი ალგორითმი: Linear Regression, Decision Tree, Random Forest და XGBoost.
ყველა ექსპერიმენტი დავარეგისტრირე MLflow-ში DagsHub-ზე. თითოეული მოდელისთვის დავლოგე შემდეგი მეტრიკები:

- `train_r2`, `val_r2`, `test_r2`
- `train_rmse`, `val_rmse`, `test_rmse`

---

###  Linear Models

შევამოწმე შემდეგი კომბინაციები:
- LinearRegression (baseline)
- Ridge: `alpha` = 0.1, 1.0, 10.0
- Lasso: `alpha` = 0.1, 1.0, 10.0

**5 საუკეთესო Linear Model შედეგი:**

| Run Name | val_r2 | test_r2 
|---|---|---|
| Ridge_a10.0 | 0.7427 | 0.7539
| Ridge_a1.0 | 0.7432 | 0.7507
| LinearRegression | 0.7431 | 0.7496
| Lasso_a10.0 | 0.7431 | 0.7499
| Lasso_a1.0 | 0.7431 | 0.7496

---

###  Decision Tree

შევამოწმე შემდეგი კომბინაციები:
- `max_depth`: 3, 5, 10, None
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4

**5 საუკეთესო Decision Tree შედეგი:**

| Run Name | val_r2 | test_r2 |
|---|---|---|
| DT_depth5_split10_leaf4 | 0.7076 | 0.7711 |
| DT_depthNone_split10_leaf5 | 0.7143 | 0.7630 |
| DT_depth10_split10_leaf4 | 0.7208 | 0.7677 |
| DT_depth5_split5_leaf2 | 0.6903 | 0.7251 |
| DT_depth3_split10_leaf4 | 0.6651 | 0.7260 |

---

###  Random Forest

შევამოწმე შემდეგი კომბინაციები:
- `n_estimators`: 100, 200, 300
- `max_depth`: 5, 10, None
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4, 5

**5 საუკეთესო Random Forest შედეგი:**

| Run Name | val_r2 | test_r2 |
|---|---|---|
| RF_n100_depth10_split2_leaf1 | 0.7849 | 0.8229 |
| RF_n100_depthNone_split2_leaf1 | 0.7825 | 0.8254 |
| RF_n300_depth10_split5_leaf2 | 0.7814 | 0.8140 |
| RF_n200_depth10_split5_leaf2 | 0.7804 | 0.8138 |
| RF_n100_depth10_split5_leaf2 | 0.7794 | 0.8161 |

Single Decision Tree-სთან შედარებით მნიშვნელოვნად გაუმჯობესდა — bagging-მა variance შეამცირა და განზოგადება გააუმჯობესა.

---

###  XGBoost

შევამოწმე შემდეგი კომბინაციები:
- `n_estimators`: 100, 200, 300, 500
- `max_depth`: 3, 5
- `learning_rate`: 0.01, 0.05, 0.1
- `subsample`: 0.8, 0.9

**5 საუკეთესო XGBoost შედეგი:**

| Run Name | val_r2 | test_r2 |
|---|---|---|
| XGB_n100_depth3_lr0.1_sub0.8 | 0.7872 | 0.8470 |
| XGB_n200_depth3_lr0.1_sub0.8 | 0.7992 | 0.8393 |
| XGB_n300_depth3_lr0.05_sub0.8 | 0.8018 | 0.8324 |
| XGB_n200_depth3_lr0.05_sub0.8 | 0.7927 | 0.8344 |
| XGB_n200_depth5_lr0.05_sub0.9 | 0.7978 | 0.8253 |

---

## საუკეთესო მოდელი

საუკეთესო მოდელი შეირჩა ყველა ექსპერიმენტის `val_r2`-ებით:

**შერჩევის კრიტერიუმი:** ყველაზე მაღალი `test_r2` სადაც `val_r2 ≈ test_r2` — ანუ როცა მოდელი პატერნებს აღიქვამს დაზეპირების ნაცვლად.

**საუკეთესო: `XGB_n100_depth3_lr0.1_sub0.8`**
- val_r2: **0.7872**
- test_r2: **0.8470**
- Kaggle Public Score (RMSLE): **0.17537**

<img width="1123" height="133" alt="image" src="https://github.com/user-attachments/assets/e645aff6-6a3d-483c-9bdc-223fa081e9d5" />

---

## MLflow ექსპერიმენტები DagsHub-ზე

dagshub : [Link](https://dagshub.com/Nestor-Dzadzamia/ML-Advanced-Regression-Techniques)

თითოეულ run-ში დაილოგა:
- ყველა ჰიპერპარამეტრი
- Train / Val / Test მეტრიკები (r2, rmse)
- დატრენინგებული მოდელის არტეფაქტი (`.skops`)

## გამოცდილება

- **Target Encoding** WOE-ს ნაცვლად regression task-ებისთვის
- **Feature Engineering** (TotalSF, TotalBath) RFE-ით ვნახეთ, რომ ორივე საბოლოო 20 feature-ში მოხვდა, რაც იმას ნიშნავს რომ ახალი სვეტები სასარგებლო იყო
- **XGBoost** linear მოდელებსა და Decision Tree-ს მნიშვნელოვნად აჯობა
- ყოველთვის **train/test split უნდა მოხდეს EDA-მდე** — წინააღმდეგ შემთხვევაში data leakage-ის რისკია
