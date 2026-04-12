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

I. სვეტების ანალიზი NA/ცარიელი მნიშვნელობებით:


<img width="1500" height="700" alt="image" src="https://github.com/user-attachments/assets/afd533c0-e8cc-454f-adb0-5799bdd8f1fe" />


**სვეტები მაღალი NA პროპორციებით:**
  *  PoolQC ~ 100%
  *  MiscFeature ~ 95%
  *  Alley ~ 95%
  *  Fence ~ 80%

მოცემულ სვეტებს (>80% NA) დავდროპავ, მეორემხრივ თუ ამ ცარიელ მნიშვნელობებს შევავსებ, დიდი რისკია overfitting-ში გადასვლის.
