# /api/app.py - VERSÃO FINAL CORRIGIDA
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# --- FUNÇÕES AUXILIARES ---
# Esta função PRECISA estar definida ANTES de carregar ou usar o pipeline
def criar_features_de_risco(df):
    """Função de engenharia de features que o modelo espera."""
    df_transformed = df.copy()
    # Garante que as colunas sejam numéricas para as operações
    for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_BIRTH']:
        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
        
    df_transformed['CREDIT_INCOME_RATIO'] = df_transformed['AMT_CREDIT'] / df_transformed['AMT_INCOME_TOTAL']
    df_transformed['ANNUITY_INCOME_RATIO'] = df_transformed['AMT_ANNUITY'] / df_transformed['AMT_INCOME_TOTAL']
    df_transformed['CREDIT_TERM'] = df_transformed['AMT_ANNUITY'] / df_transformed['AMT_CREDIT']
    df_transformed['EMPLOYED_BIRTH_RATIO'] = df_transformed['DAYS_EMPLOYED'] / df_transformed['DAYS_BIRTH']
    df_transformed.replace([np.inf, -np.inf], 0, inplace=True) # Troca inf por 0
    df_transformed.fillna(0, inplace=True) # Troca NaN por 0
    return df_transformed

# --- O restante das suas funções auxiliares (sem alterações) ---
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
    if credito <= 1000: score += 300
    if tempo_emprego_anos > 10: score += 100
    if renda > 20000: score += 100
    return score, motivos

def gerar_motivos_ml(data):
    motivos = []
    renda = float(data.get('AMT_INCOME_TOTAL', 0))
    parcela = float(data.get('AMT_ANNUITY', 0))
    credito = float(data.get('AMT_CREDIT', 1))
    if renda > 0 and (parcela / renda) > 0.4: motivos.append("O comprometimento de renda (parcela/renda) foi considerado alto pelo modelo.")
    if credito > 0 and (credito / renda) > 8: motivos.append("O valor de crédito solicitado é alto em comparação com sua renda.")
    if -data.get('DAYS_EMPLOYED', 0) / 365 < 3: motivos.append("A estabilidade no emprego foi um fator de atenção para o modelo.")
    if not motivos: motivos.append("A combinação de suas informações resultou em um score de risco elevado, segundo nossa análise preditiva.")
    return motivos

# --- CONFIGURAÇÃO DO APP E CARREGAMENTO DO MODELO ---
app = Flask(__name__)
CORS(app)

# Carregamento do modelo
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'modelo_simulador_web.joblib')
    pipeline = joblib.load(model_path)
    print("Pipeline do modelo carregado com sucesso!")
except Exception as e:
    print(f"ERRO CRÍTICO AO CARREGAR O MODELO: {e}")
    pipeline = None

# --- ROTA PRINCIPAL DA API ---
@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Modelo não foi carregado. Verifique os logs do servidor.'}), 500

    try:
        data = request.get_json()
        
        # --- BLOCO DE VALIDAÇÃO (Sem alterações) ---
        try:
            idade_anos = -float(data.get('DAYS_BIRTH', 0)) / 365
            # ... (seu bloco de validação completo aqui) ...
            filhos = int(data.get('CNT_CHILDREN', 0))
            tempo_trabalho_anos = -float(data.get('DAYS_EMPLOYED', 0)) / 365
            renda = float(data.get('AMT_INCOME_TOTAL', 0))
            credito = float(data.get('AMT_CREDIT', 0))
            parcela = float(data.get('AMT_ANNUITY', 0))
            valor_bem = float(data.get('AMT_GOODS_PRICE', 0))
            if not (18 <= idade_anos <= 100): return jsonify({'error': 'Idade fora do intervalo permitido (18-100 anos).'}), 400
            if not (0 <= filhos <= 30): return jsonify({'error': 'Número de filhos fora do intervalo permitido (0-30).'}), 400
            if not (0 <= tempo_trabalho_anos <= 60): return jsonify({'error': 'Tempo de trabalho fora do intervalo permitido (0-60 anos).'}), 400
            max_tempo_trabalho = idade_anos - 14
            if idade_anos > 14 and tempo_trabalho_anos > max_tempo_trabalho:
                mensagem_erro = f"Para {int(idade_anos)} anos de idade, o tempo de trabalho não pode ser maior que {int(max_tempo_trabalho)} anos."
                return jsonify({'error': mensagem_erro}), 400
            if not (0 < renda <= 2000000): return jsonify({'error': 'Renda mensal fora do intervalo permitido (R$ 1 a R$ 2.000.000).'}), 400
            if not (0 < credito <= 10000000): return jsonify({'error': 'Valor de crédito solicitado fora do intervalo permitido (até R$ 10.000.000).'}), 400
            if not (0 < parcela <= 500000): return jsonify({'error': 'Valor da parcela fora do intervalo permitido (até R$ 500.000).'}), 400
            if not (0 < valor_bem <= 10000000): return jsonify({'error': 'Valor do bem adquirido fora do intervalo permitido (até R$ 10.000.000).'}), 400
        except (ValueError, TypeError):
             return jsonify({'error': 'Dados de entrada inválidos. Verifique os tipos de dados.'}), 400

        # --- LÓGICA HÍBRIDA ---
        scorecard_base, motivos_scorecard = calcular_scorecard_e_motivos(data)
        
        score_final_prob, motivos_recusa = 0, []
        if scorecard_base < 400:
            score_final_prob = 0.75 + (400 - scorecard_base) / 1000.0
            motivos_recusa = motivos_scorecard
        elif scorecard_base > 650:
            score_final_prob = 0.10 - (scorecard_base - 650) / 1000.0
        else:
            # MUDANÇA CRÍTICA AQUI:
            # 1. Cria o DataFrame
            input_df = pd.DataFrame([data])
            # 2. Chama a função de transformação de features explicitamente
            transformed_df = criar_features_de_risco(input_df)
            # 3. Faz a predição no DataFrame transformado
            score_final_prob = pipeline.predict_proba(transformed_df)[:, 1][0]
            if score_final_prob > 0.5: motivos_recusa = gerar_motivos_ml(data)

        score_final_seguro = max(0.01, min(0.99, score_final_prob))
        score_final_formatado = converter_prob_para_score(score_final_seguro)

        return jsonify({'score_formatado': score_final_formatado, 'motivos_recusa': motivos_recusa})
    except Exception as e:
        return jsonify({'error': f'Erro durante a predição: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)