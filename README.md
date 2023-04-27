# AutoML_Bachelors_thesis
## Analysis of machine learning models used in the process of building model ensembles for the regression task.  
<p align = "right">
<img src="mini_mini_logo.png" align="middle" width="150"/> <img src="delfis_logo.png" align="middle" width="150"/>
</p>

---
**GOAL**

We conduct an expirement in order to gather data about model ensembles created by AutoML frameworks for the regression task. We focus on [AutoGluon](https://auto.gluon.ai/stable/index.html) and [AutoSklearn](https://automl.github.io/auto-sklearn/master/) and aim to achieve conclusions about the form of the before mentioned ensembles. The expirement is carried out on a particular group of data sets from [OpenML](https://www.openml.org/).


---

---
**CONTENTS**

This repository is a collection of functions used throughout to thesis.  The **run_autosklearn_reg** and **run_autogluon** functions work directly with both frameworks in order to fit their best ensembles. **run_regression** is used to run the mentioned functions across a certain parameter grid in order to compare the different results. Finally we use **extract_data** to gather the sepicifics of the trained emsebles into an elegant format. 

---

