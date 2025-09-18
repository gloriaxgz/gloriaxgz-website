# /api/app.py - VERSÃO FINAL COM DOWNLOAD DO HUGGING FACE HUB
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from huggingface_hub import hf_hub_download # <-- NOVA IMPORTAÇÃO

# --- FUNÇÃO DE CRIAÇÃO DE FEATURES (IDÊNTICA À DO NOTEBOOK DE TREINO) ---
def criar_features_de_risco(df):
    """Função de engenharia de features que o modelo espera."""
    df_transformed = df.copy()
    for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_BIRTH']:
        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
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

# --- FUNÇÕES AUXILIARES DE SCORE ---
def converter_prob_para_score(probabilidade_inadimplencia):
    """Converte a probabilidade (0-1) para um score (100-1000)."""
    score = 1000 - (probabilidade_inadimplencia * 900)
    return int(round(score))

def calcular_scorecard_e_motivos(data):
    """Calcula o score de regras e já retorna os motivos de recusa associados."""
    score, motivos = 500, []
    # ... (Sua lógica de scorecard aqui, sem alterações)
    return score, motivos

def gerar_motivos_ml(data):
    """Gera motivos de recusa simplificados para a 'zona cinzenta' do ML."""
    motivos = []
    # ... (Sua lógica de motivos de ML aqui, sem alterações)
    return motivos

# --- CONFIGURAÇÃO DO APP E CARREGAMENTO DO MODELO DO HUB ---
app = Flask(__name__)
CORS(app)

pipeline = None
try:
    # --- MUDANÇA CRÍTICA: Carrega o modelo do Hugging Face Hub ---
    print("Baixando o modelo do Hugging Face Hub...")
    
    # IMPORTANTE: Substitua 'SEU_USUARIO_HF/NOME_DO_SEU_MODELO' pelo ID do seu repositório de modelo
    # Você encontra esse ID no topo da página do seu modelo. Ex: 'gloriaxgz/modelo-credito'
    MODEL_REPO_ID = "gloriaxgz/modelo-scorecard" 
    MODEL_FILENAME = "modelo_final_deploy.joblib"

    # Baixa o arquivo e obtém o caminho local para ele
    cached_model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    
    print(f"Modelo baixado para: {cached_model_path}")
    
    # Carrega o pipeline do arquivo baixado
    pipeline = joblib.load(cached_model_path)
    
    print("✅ Pipeline de produção carregado com sucesso do Hugging Face Hub!")

except Exception as e:
    print(f"❌ ERRO CRÍTICO AO CARREGAR O MODELO: {e}")

# --- ROTA PRINCIPAL DA API ---
@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Modelo não carregado. Verifique os logs do servidor.'}), 500

    try:
        data = request.get_json()
        
        # --- BLOCO DE VALIDAÇÃO DOS DADOS DE ENTRADA (sem alterações) ---
        # ... (Sua lógica de validação aqui) ...

        # --- LÓGICA HÍBRIDA (sem alterações) ---
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

        return jsonify({'score_formatado': score_final_formatado, 'motivos_recusa': motivos_recusa})
    except Exception as e:
        print(f"ERRO NA PREDIÇÃO: {e}")
        return jsonify({'error': f'Erro durante a predição: {str(e)}'}), 500

# --- EXECUÇÃO LOCAL ---
if __name__ == '__main__':
    # A porta 8080 é frequentemente usada por padrão em serviços como o Hugging Face
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))