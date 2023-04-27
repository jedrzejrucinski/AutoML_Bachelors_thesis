def get_autosklearn_reg(X,y, ensembling_size, per_time_budget, overall_budget_time, data_id,
                     model_set = None):
  
    model = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task = overall_budget_time,
        ensemble_size = ensembling_size,
        per_run_time_limit = per_time_budget,
        include = model_set,
        n_jobs =  -1,
        memory_limit = None
    )


    size = X.shape

    model.fit(X, y)
    splitByComma=model.sprint_statistics().split('\n')
   
    
    weighted_models_config_id = model.leaderboard(detailed = True).config_id.to_list()
    weighted_models_weights = model.leaderboard(detailed = True).ensemble_weight.to_list()

    dikt = {}
    modele = model.cv_results_['status']
    j = 0
    for i in weighted_models_config_id:
        if modele[i-1] == 'Success':
            klucze = list(model.cv_results_['params'][i-1].keys())
            dikt[f"{j}_" + f"{model.cv_results_['params'][i-1]['regressor:__choice__']}"] = {}
        for k in range(len(klucze)):
            dikt[f"{j}_" + f"{model.cv_results_['params'][i-1]['regressor:__choice__']}"][klucze[k]] = model.cv_results_['params'][i-1][klucze[k]]
        dikt[f"{j}_" + f"{model.cv_results_['params'][i-1]['regressor:__choice__']}"]['val_score'] = model.cv_results_['mean_test_score'][i-1]
        dikt[f"{j}_" + f"{model.cv_results_['params'][i-1]['regressor:__choice__']}"]['fit_time'] = model.cv_results_['mean_fit_time'][i-1]
        dikt[f"{j}_" + f"{model.cv_results_['params'][i-1]['regressor:__choice__']}"]['model_weight'] = weighted_models_weights[weighted_models_config_id.index(i)]
        j += 1


    dictionary = {'id': data_id,
            'size': size,
            "algorythm": "Autosklearn",
            'ensembling_size': ensembling_size,
            'per_time_budget' : per_time_budget,
            'overall_budget_time' : overall_budget_time,
            'model_set' : model_set,
            'metric' : splitByComma[2].replace(' Metric: ', ''),
            'best validation score' : splitByComma[3].replace(' Best validation score: ', ''),
            'models' : dikt
            }
    return dictionary
