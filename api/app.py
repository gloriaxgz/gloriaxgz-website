import os
import joblib
import pandas as pd
import numpy as np
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)

def criar_features_de_risco(df):
    df_transformed = df.copy()
    numeric_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
    for col in numeric_cols:
        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
    df_transformed[numeric_cols] = df_transformed[numeric_cols].fillna(0)
    df_transformed['AMT_INCOME_TOTAL'] = df_transformed['AMT_INCOME_TOTAL'].replace(0, 1)
    df_transformed['AMT_CREDIT'] = df_transformed['AMT_CREDIT'].replace(0, 1)
    df_transformed['DAYS_BIRTH'] = df_transformed['DAYS_BIRTH'].replace(0, -1)
    df_transformed['CREDIT_INCOME_RATIO'] = df_transformed['AMT_CREDIT'] / df_transformed['AMT_INCOME_TOTAL']
    df_transformed['ANNUITY_INCOME_RATIO'] = df_transformed['AMT_ANNUITY'] / df_transformed['AMT_INCOME_TOTAL']
    df_transformed['CREDIT_TERM'] = df_transformed['AMT_ANNUITY'] / df_transformed['AMT_CREDIT']
    df_transformed['EMPLOYED_BIRTH_RATIO'] = df_transformed['DAYS_EMPLOYED'] / df_transformed['DAYS_BIRTH']
    df_transformed.replace([np.inf, -np.inf], 0, inplace=True)
    df_transformed.fillna(0, inplace=True)
    return df_transformed

def converter_prob_para_score(probabilidade_inadimplencia):
    score = 1000 - (probabilidade_inadimplencia * 900)
    return int(round(score))

def calcular_scorecard_e_motivos(data):
    score, motivos = 500, []
    renda = float(data.get('AMT_INCOME_TOTAL', 0))
    credito = float(data.get('AMT_CREDIT', 1))
    parcela = float(data.get('AMT_ANNUITY', 0))
    idade_anos = -data.get('DAYS_BIRTH', 0) / 365
    tempo_emprego_anos = -data.get('DAYS_EMPLOYED', 0) / 365
    if renda > 0 and (parcela / renda) > 0.6:
        score -= 400
        motivos.append("O valor da parcela compromete uma parte muito grande da sua renda mensal.")
    if idade_anos < 22:
        score -= 150
        motivos.append("A análise de perfis mais jovens indica um risco maior para este tipo de crédito.")
    if tempo_emprego_anos < 1:
        score -= 150
        motivos.append("O pouco tempo no emprego atual é um fator de risco.")
    return score, motivos

def gerar_motivos_ml(data):
    motivos = []
    renda = float(data.get('AMT_INCOME_TOTAL', 0))
    parcela = float(data.get('AMT_ANNUITY', 0))
    if renda > 0 and (parcela / renda) > 0.4:
        motivos.append("O comprometimento de renda (parcela/renda) foi considerado alto pelo modelo.")
    if -data.get('DAYS_EMPLOYED', 0) / 365 < 3:
        motivos.append("A estabilidade no emprego foi um fator de atenção para o modelo.")
    if not motivos:
        motivos.append("A combinação de suas informações resultou em um score de risco elevado.")
    return motivos

pipeline = None
try:
    logging.info("Iniciando o carregamento do modelo do Hugging Face Hub...")
    
    MODEL_REPO_ID = "gloriaxgz/modelo-credito" 
    MODEL_FILENAME = "modelo_final_deploy.joblib"

    cached_model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME
    )
    logging.info(f"Modelo descarregado com sucesso para: {cached_model_path}")

    pipeline = joblib.load(cached_model_path)
    logging.info("✅ Pipeline de produção carregado e pronto!")

except Exception as e:
    logging.error("❌ ERRO CRÍTICO AO CARREGAR O MODELO:")
    logging.error(traceback.format_exc())

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        logging.error("O modelo não foi carregado. A abortar a requisição.")
        return jsonify({'error': 'Modelo não carregado. Verifique os logs do servidor.'}), 500
    try:
        data = request.get_json()
        logging.info(f"Requisição recebida: {data}")

        idade_anos = -float(data.get('DAYS_BIRTH', 0)) / 365
        if not (18 <= idade_anos <= 100):
            return jsonify({'error': 'Idade fora do intervalo permitido (18-100 anos).'}), 400
        
        scorecard_base, motivos_scorecard = calcular_scorecard_e_motivos(data)
        score_final_prob, motivos_recusa = 0, []

        if scorecard_base < 400:
            score_final_prob = 0.75 + (400 - scorecard_base) / 1000.0
            motivos_recusa = motivos_scorecard
        elif scorecard_base > 650:
            score_final_prob = 0.10 - (scorecard_base - 650) / 1000.0
        else:
            input_df = pd.DataFrame([data])
            transformed_df = criar_features_de_risco(input_df)
            
            prediction_output = pipeline.predict_proba(transformed_df)
            
            if prediction_output.ndim == 2:
                score_final_prob = float(prediction_output[0, 1])
            else:
                score_final_prob = float(prediction_output[0])

            if score_final_prob > 0.5:
                motivos_recusa = gerar_motivos_ml(data)

        score_final_seguro = max(0.01, min(0.99, score_final_prob))
        score_final_formatado = converter_prob_para_score(score_final_seguro)
        
        logging.info(f"Score final calculado: {score_final_formatado}")
        return jsonify({'score_formatado': score_final_formatado, 'motivos_recusa': motivos_recusa})

    except Exception as e:
        logging.error("❌ ERRO NA PREDIÇÃO:")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Erro durante a predição: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
