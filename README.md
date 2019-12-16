### btc_codebase
A code base for testing btc trade algorithm

---
##### train mode:<br>
Model trained with holding_amount_data and btc price, algorithm selected from **LinearRegression**, **SVR**, **KNeighborsRegressor**
```
python btc_model.py \
	--grid_search False \
	--read_path './data' \
	--save_path './output' \
	--train

# given "btc_price, 持有量100-100, etc.." in data folder
# return best model and store in output folder

return example:

LinearRegression       R-squared: -10.418
best_score: -10.418
SVR                    R-squared: -4.444
best_score: -4.444
KNeighborsRegressor    R-squared: 0.517
best_score: 0.517
trend simultaneous acc: 0.979
save model in paht KNeighborsRegressor

```

##### eval mode:
eval model without training, assuming there is model existed in *output* directory. Prediction will be appened to *data/pred.csv*.
```
python btc_model.py \
	--read_path './data' \
	--save_path './output' \
	--t_next
# given "持有量100-1000-test, etc.." in data folder, test tag must be included in file name
# return T+1/T+N prediction(make sure there is a model get trained before)

T+1 example:
current time: 2019-11-27 07:37:04
next time: 2019-11-27 10:37:04
price prediction: 7771.996548204

```
#### Todo:
- [x] Eval Mode: test trained model, return prediction with given data [2019.12.16]
- [ ] Plotting: draw price and prediction in real time
- [ ] More auxiliary data: e.g. mood data
- [ ] Model's Parameter Search: random search, bayesian
- [ ] Advanced Model: RNN coming
- [ ] Data quality checking
- [x] Use logging to replace print
