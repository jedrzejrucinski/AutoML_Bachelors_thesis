def extract_from_json(path):

    import os,json
    import pandas as pd
    import numpy as np
    from fuzzywuzzy import fuzz
    import time

    tabela = {
    'dataset_id' : [],
    'dataset_n' : [],
    'dataset_p' : [],


    'ensembling_size' : [],
    'per_time_budget' : [],
    'overall_budget_time' : [],
    'model_set' : [],
    
    'framework' : [],
    'ensemble_id' : [],
    'original_model_name': [],
    'model': [],
    'parameters' : [],
    'weight' : [],
    'fit_time' : [],
    'val_score' : [],
    'preprocessing' :[],
    }  
    
    modele_autosklearn = [
        'adaboost',
        'bernoulli_naive_bayes',
        'decision_tree',
        'extra_trees',
        'gaussian_naive_bayes',
        'gradient_boosting',
        'k_nearest_neighbors',
        'lda',
        'liblinear_svc',
        'libsvm_svc',
        'neural_network',
        'multinomial_naive_bayes',
        'passive_aggressive',
        'qda',
        'random_forest',
        'sgd',
        'ard_regression',
        'gaussian_process' 
    ]
    modele_autogluon = [
        'abstractmodel',
        'light_gbm',
        'catboost',
        'xgboost',
        'ranrandom_forest',
        'extra_trees',
        'k_nearest_neighbors',
        'linear_model.logistic_regression',
        'neural_network',
        'fastai_v1_neural_network',
        'vowpal_wabbit'
    ]

    filenames = [filename for filename in os.listdir(path)]

    l = 1
    for file in filenames:
        with open(os.path.join(path,file),'r') as results:
            data = json.load(results)

        if data['algorythm'] == 'Autosklearn':
            iter = list(data['models'].keys())
            for j in range(len(iter)):
                tabela['original_model_name'].append(iter[j])
                tabela['ensembling_size'].append(data['ensembling_size']),
                tabela['per_time_budget'].append(data['per_time_budget']),
                tabela['overall_budget_time'].append(data['overall_budget_time']),
                tabela['model_set'].append(data['model_set']),
                tabela['dataset_id'].append(data['id'])
                tabela['dataset_n'].append(data['size'][0])
                tabela['dataset_p'].append(data['size'][1])
                tabela['framework'].append(data['algorythm'])
                tabela['ensemble_id'].append(f'{l}_{time.strftime("%Y%m%d-%H%M%S")}')


                badziewia = list(data['models'][iter[j]].keys())
                parameters = {}
                for b in range(len(badziewia)):
                    if (badziewia[b].startswith('regressor') and not badziewia[b].startswith('regressor:__choice__')) or (badziewia[b].startswith('classifier') and not badziewia[b].startswith('classifier:__choice__')):
                 
                        
                        while not badziewia[b].startswith('data_preprocessor') and not badziewia[b].startswith('val_score'):
                            x = badziewia[b].split(':')
                            parameters[x[2]] = data['models'][iter[j]][badziewia[b]]
                            b += 1
                        break
                if parameters == {}:
                    parameters['model_params'] = 'No model parameters'
                tabela['parameters'].append(parameters)                
                        
                
                tabela['weight'].append(data['models'][iter[j]]['model_weight'])
                tabela['fit_time'].append(data['models'][iter[j]]['fit_time'])
                tabela['val_score'].append(data['models'][iter[j]]['val_score'])
                tabela['preprocessing'].append(0)

                if iter[j][2:] == 'mlp':
                    tabela['model'].append('neural_network')
                else:
                    df_dobre = pd.DataFrame({'nazwy_po' : modele_autosklearn})
                    df_original = pd.DataFrame({'originalna_nazwa' : [iter[j]]})

                    df_original['dummy'] = True
                    df_dobre['dummy'] = True

                    df_fuzzy = pd.merge(df_original, df_dobre, on = 'dummy')

                    df_fuzzy.drop('dummy', axis = 1, inplace = True)
                    df_fuzzy['Token_Set_Ratio'] = df_fuzzy[['originalna_nazwa','nazwy_po']].apply(lambda x:fuzz.token_set_ratio(x.originalna_nazwa, x.nazwy_po), axis=1)
                    choice = df_fuzzy.loc[df_fuzzy['Token_Set_Ratio'].idxmax()]
                    tabela['model'].append(choice.nazwy_po)


            
        if data['algorythm'] == 'AutoGluon':
            iter = list(data['best ensemble'].keys())
            for j in range(len(iter)):
                tabela['original_model_name'].append(iter[j])
                tabela['ensembling_size'].append(data['ensembling_size']),
                tabela['per_time_budget'].append(data['per_time_budget']),
                tabela['overall_budget_time'].append(data['overall_budget_time']),
                tabela['model_set'].append(data['model_set']),
                tabela['dataset_id'].append(data['data_id'])
                tabela['dataset_n'].append(data['size'][0])
                tabela['dataset_p'].append(data['size'][1])
                tabela['framework'].append(data['algorythm'])
                tabela['ensemble_id'].append(f'{l}_{time.strftime("%Y%m%d-%H%M%S")}')
                tabela['parameters'].append(data['best ensemble'][iter[j]]['general_hyperparameters'])
                tabela['weight'].append(data['best ensemble'][iter[j]]['model_weight'])
                tabela['fit_time'].append(data['best ensemble'][iter[j]]['fit_time'])
                tabela['val_score'].append(data['best ensemble'][iter[j]]['val_score'])
                tabela['preprocessing'].append(0)
                

                df_dobre = pd.DataFrame({'nazwy_po' : modele_autogluon})
                df_original = pd.DataFrame({'originalna_nazwa' : [iter[j]]})

                df_original['dummy'] = True
                df_dobre['dummy'] = True

                df_fuzzy = pd.merge(df_original, df_dobre, on = 'dummy')

                df_fuzzy.drop('dummy', axis = 1, inplace = True)
                df_fuzzy['Token_Set_Ratio'] = df_fuzzy[['originalna_nazwa','nazwy_po']].apply(lambda x:fuzz.token_set_ratio(x.originalna_nazwa, x.nazwy_po), axis=1)
                choice = df_fuzzy.loc[df_fuzzy['Token_Set_Ratio'].idxmax()]
                tabela['model'].append(choice.nazwy_po)
        l += 1     

    tabela_pd = pd.DataFrame(tabela)
    
    return tabela_pd  
