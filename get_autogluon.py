def get_autogluon(X,y, ensembling_size, overall_budget_time, per_time_budget, data_id, problem_type, model_set = None,path = '/workspace/funkcje lic/AutogluonModels'):

    class mydict(dict):
        def __getitem__(self, value):
            keys = [k for k in self.keys() if value in k]
            key = keys[0] if keys else None
            return self.get(key)
    X['target'] = y

    from autogluon.tabular import TabularDataset, TabularPredictor
    train_data = TabularDataset(X)
    label = 'target'
    predictor = TabularPredictor(label = label, problem_type= problem_type, path = path)
    predictor.fit(
        train_data = train_data,
        time_limit = overall_budget_time,
        num_bag_folds = ensembling_size,
        infer_limit = per_time_budget/len(X),
                                           
        ag_args_ensemble={'use_child_oof': False},
        )


    inf = predictor.info() 
    my_info = mydict(inf['model_info'])
    best_ensemble = my_info[predictor.get_model_best()]
    models = best_ensemble['children_info']['S1F1']['model_weights'].keys()
    models = list(models)
    
    parts = {}

  
    for i in models:
        parts[f'{i}'] = {} 
        parts[f'{i}']['val_score'] = my_info[f'{i}']['val_score']
        parts[f'{i}']['fit_time'] = my_info[f'{i}']['fit_time']
        parts[f'{i}']['predict_time'] = my_info[f'{i}']['predict_time']
        parts[f'{i}']['model_weight'] = best_ensemble['children_info']['S1F1']['model_weights'][f'{i}']
        parts[f'{i}']['general_hyperparameters'] = inf['model_info'][f'{i}']['children_info']['S1F1']['hyperparameters']

        children = inf['model_info'][f'{i}']['children_info'].keys()
        children = list(children)
        for j in children:
            parts[f'{i}'][f'{j}']= inf['model_info'][f'{i}']['children_info'][f'{j}']['hyperparameters']
            parts[f'{i}'][f'{j}']['hyperparameters_fit'] = inf['model_info'][f'{i}']['children_info'][f'{j}']['hyperparameters_fit']

    result = {
                'data_id': data_id,
                'algorythm':'AutoGluon',
                'ensembling_size': ensembling_size,
                'per_time_budget' : per_time_budget,
                'overall_budget_time' : overall_budget_time,
                'model_set' : model_set,
                'size': X.shape,
                'metric': my_info[predictor.get_model_best()]['eval_metric'],
                'best ensemble': parts,
                'best ensemble validation score': inf['best_model_score_val'],
                'total fit time': inf['time_fit_total']
            }
    return result
