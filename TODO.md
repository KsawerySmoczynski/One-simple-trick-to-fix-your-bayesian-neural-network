### 1. Refactor
* Draw weights or configure them in separate script, also save histogram. Should have functionality of current evaluate_weights_2d.py
* Run 1d search in one script, add functionality do set ranges for certain weights
* Run finer 1d evaluation over the interval provided from previous script
* Perform weights-wise 2d evaluations over the intervals provided in first point
* Generalize to support arbitrary eval loop -> eval loop should be contained in scr and case dependent:
    * Separate ones for: classification (images), regression (simple datasets), forecasting etc.

### 2. Create baselines for simple regression datasets

### 3. Create baselines for time series forecasting
* Use [nbeats model from pytorch forecasting package](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/ar.html)
