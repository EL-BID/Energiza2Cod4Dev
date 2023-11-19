import joblib

def make_predictions(df_infernce,file_cols_for_model,file_stacking_model_name,inference_version,cols_to_send,file_name_inference_to_send):
    cols_for_model = joblib.load(file_cols_for_model)
    combination_final = joblib.load(file_stacking_model_name)

    y_pred_inf_comb = combination_final.predict_proba(df_infernce[cols_for_model])[:,1]
    df_infernce['pred_score']= y_pred_inf_comb
    df_infernce['id_seleccion'] = inference_version
    df_infernce.sort_values('pred_score',ascending=False, inplace=True)
    df_infernce[cols_to_send].to_csv(file_name_inference_to_send, index=False)