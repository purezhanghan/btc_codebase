### btc_codebase
A code base for testing btc trade algorithm

---
##### train mode:<br>
train with holding_amount_data and btc price, model selected from **LinearRegression**, **SVR**, **KNeighborsRegressor**
```
python btc_model.py \
	--grid_search False \
	--read_path './data' \
	--save_path './output' \
	--train
```

##### eval mode:
eval model without training, assuming there is model existed in *output* directory
```
python btc_model.py \
	-- save_path './output' \
	--train False
```
#### Todo:
- [ ] Eval Mode: test trained model, return prediction with given data
- [ ] Plotting: draw price and prediction in real time
- [ ] More auxiliary data: e.g. mood data
- [ ] Model's Parameter Search: random search, bayesian
- [ ] Advanced Model: RNN coming
