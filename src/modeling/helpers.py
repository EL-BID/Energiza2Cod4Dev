import joblib

def save_model(model,file_name,is_pkl=True):
    if is_pkl:
        joblib.dump(model, file_name)
    else:
        model.save(file_name)