from all_get_functions import *

def run_regression(ids, n, overall_bugdet_time, per_run_budget, ensemble_size, path, ag_path,framework = ['autosklearn','autogluon']):
    for _ in range(n):
        for fw in framework:
            for id in ids:
                X,y = fetch_openml(data_id = id, return_X_y=True) 
                for obt in overall_bugdet_time:
                    for prb in per_run_budget:
                        for es in ensemble_size:     
                            if fw == "autosklearn":
                                results = get_autosklearn_reg(X,y, es, prb, obt, id, model_set = None)
                            if fw == "autogluon":
                                results = get_autogluon(X,y,es, obt, prb, id, problem_type = 'regression',model_set = None,path = ag_path)

                            with open(f'{path}/{id}_results_{time.strftime("%Y%m%d-%H%M%S")}.json', 'a') as file_object:  #open the file in write mode
                                json.dump(results, file_object, ensure_ascii= False, indent = 4)   # json.dump() function to stores the set of numbers in numbers.json file  
                dir = ag_path
                for files in os.listdir(dir):
                    sciezka = os.path.join(dir,files)
                    try:
                        shutil.rmtree(sciezka)
                    except OSError:
                        os.remove(sciezka)
