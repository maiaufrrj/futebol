# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:23:55 2022

@author: joao.oliveira
"""

import pandas as pd
import numpy as np
#import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
#from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import config

cutoff=0.90

lista_probabilidades = []
lista_precisao = []
lista_odd_alvo = []
lista_targets = []

lista_ensaios = config.lista_ensaios

for i in range(len(lista_ensaios)):
    print('ensaio: ', i+1)
    test_size = lista_ensaios[i][0]
    tipo = lista_ensaios[i][1]
    ALVO_PRIMARIO = lista_ensaios[i][2]
    ALVO_SECUNDARIO = lista_ensaios[i][3]

    def sum_goals(sheet_name='list', filename = 'dados_futebol.xlsx', sheet_validacao='test'):
        data = pd.read_excel(open(filename, 'rb'), sheet_name=sheet_name)
        data = data[data['time_casa'].notna()]
        data.total_gols = data.placar_final_casa+data.placar_final_fora
        data = data.total_gols
        data_val = pd.read_excel(open(filename, 'rb'), sheet_name=sheet_validacao)
        data_val = data_val[data_val['gols_casa'].notna()]
        data_val.total_gols = data_val.gols_casa+data_val.gols_visitante
        data_val = data_val.total_gols
        data = pd.concat([data,data_val])
        return data
    
    df_results = sum_goals()
    
    #stats_alvo1
    def load_data(validacao=False, sheet_name='stats_alvo1', filename = 'dados_futebol.xlsx', sheet_validacao='test'):    
        '''carregar dados e criar features'''
        df = pd.read_excel(open(filename, 'rb'), sheet_name=sheet_name)
        df = df[df['id'].notna()]
        novas_features=df.copy()
        novas_features['x1'] = abs(df['gols_casa'] - df['gols_visitante'])
        novas_features['x2'] = abs(df['tentativas_gol_casa'] - df['tentativas_gol_visitante'])
        novas_features['x3'] = abs(df['finalizacoes_casa'] - df['finalizacoes_visitante'])
        novas_features['x4'] = abs(df['chutes_fora_casa'] - df['chutes_fora_visitante'])
        novas_features['x5'] = abs(df['chutes_bloqueados_casa'] - df['chutes_bloqueados_visitante'])
        novas_features['x6'] = abs(df['falta_cobrada_casa'] - df['falta_cobrada_visitante'])
        novas_features['x7'] = abs(df['escanteios_casa'] - df['escanteios_visitante'])
        novas_features['x8'] = abs(df['impedimentos_casa'] - df['impedimentos_visitante'])
        novas_features['x9'] = abs(df['laterais_cobrados_casa'] - df['laterais_cobrados_visitante'])
        novas_features['x10'] = abs(df['defesas_goleiro_casa'] - df['defesas_goleiro_visitante'])
        novas_features['x11'] = abs(df['faltas_casa'] - df['faltas_visitante'])
        novas_features['x12'] = abs(df['cartoes_vermelhos_casa'] - df['cartoes_vermelhos_visitante'])
        novas_features['x13'] = abs(df['cartoes_amarelos_casa'] - df['cartoes_amarelos_visitante'])
        novas_features['x14'] = abs(df['total_passes_casa'] - df['total_passes_visitante'])
        novas_features['x15'] = abs(df['passes_completados_casa'] - df['passes_completados_visitante'])
        novas_features['x16'] = abs(df['desarme_casa'] - df['desarme_visitante'])
        novas_features['x17'] = abs(df['ataques_casa'] - df['ataques_visitante'])
        novas_features['x18'] = abs(df['ataques_perigosos_casa'] - df['ataques_perigosos_visitante'])
        novas_features['x19'] = novas_features.x1+((novas_features.x2)**2)+novas_features.x3+novas_features.x7 + novas_features.x8 + novas_features.x14 + (novas_features.x15)**2+novas_features.x16+novas_features.x17+(novas_features.x18)**2
        novas_features['x20'] = abs(df.casa_ganhando)
        #novas_features['x21'] = abs(df['ataques_perigosos_casa'] + df['ataques_perigosos_visitante'])+abs(df['ataques_casa'] + df['ataques_visitante'])+abs(df['finalizacoes_casa'] - df['finalizacoes_visitante'])
        novas_features['soma_gols_primeiro_tempo'] = df['gols_casa']+df['gols_visitante']
        novas_features=novas_features.iloc[:,-21:]
        #df = df[df['id']!=id_remover_do_treino]
        df.id = df.id.astype(int)
        df.set_index(df.id,inplace=True)
        df = df.iloc[:df.target.count(),1:]
        df.target = df.target.astype(int)
        df.casa_ganhando = df.casa_ganhando.astype(int)
        #verificando a features q poderão ser usadas, conforme ultimos dados de validação
        if validacao==False:
            df_valid = pd.read_excel(open(filename, 'rb'), sheet_name=sheet_validacao)
            df_valid = df_valid.dropna(axis=1)
            features_names = list(df_valid)
            features_names.remove('id')
            df=df[features_names]
            df=pd.concat([df,novas_features], axis=1)
        else:
            df=pd.concat([df,novas_features],axis=1)
            df = df.dropna(axis=1)
        return df
    
    df_valid=load_data(validacao=True, sheet_name='test', filename = 'dados_futebol.xlsx')
    df=load_data()[list(df_valid)]#.dropna(axis=0)
    column_means = df.mean()
    df = df.fillna(column_means)
    n_linhas_valid = 1 #df_valid.shape[0]
    df=pd.concat([df,df_valid],axis=0)
    
    #target calculado aqui
    def calculate_target(resultados=df_results, data=df, tipo='menor'):
        data['target'] = (resultados)-(data.gols_casa+data.gols_visitante)
        if tipo=='menor':
            data['target'] = np.where(data['target']>=ALVO_PRIMARIO,0,1)
        elif tipo=='maior':
            data['target'] = np.where(data['target']>=ALVO_PRIMARIO,1,0)
        else:
            data['target'] = np.where((data['target']>=ALVO_PRIMARIO)&(data['target']<=ALVO_SECUNDARIO),1,0)
        return data
    
    df=calculate_target(tipo=tipo)
    
    x = df.iloc[:,1:]
    start_letter = 'x'
    nomes_colunas = list(x)
    subset = [x for x in nomes_colunas if x.startswith(start_letter)]
    
    # Filtrando os missing values
    def fatorar_qcut(data=x, subset=subset):
        data_filtrado = data
        for i in range(data_filtrado.shape[1]):
            corte = 9
            name=str(list(data_filtrado)[i])
            try:
                data_filtrado[name] = pd.qcut(data_filtrado[name], corte, labels = False)
            except:
                while corte >= 3:
                    corte=corte-1
                    try:
                        data_filtrado[name] = pd.qcut(data_filtrado[name], corte, labels = False)
                    except:
                        pass
        return data_filtrado
    
    x=fatorar_qcut()
    #ignore_features = ['x7','x12','x13'] #baseado em shap values
    #for i in range(len(ignore_features)):
    #    try:
    #        x=x.drop(ignore_features, axis=1)
    #    except:
    #        pass
        
    y = df.target.iloc[:-n_linhas_valid]
    
    x_valid = x.iloc[-n_linhas_valid:,:]
    x=x.iloc[:-n_linhas_valid,:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    
    #oversample = BorderlineSMOTE(m_neighbors = 5, random_state=0)
    #x_train, y_train = oversample.fit_resample(x_train, y_train)
    
    sm = SMOTE(random_state=0)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    
    df_train = pd.concat([y_train,x_train], axis=1)
    
    #model = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=10)
    #from sklearn.ensemble import GradientBoostingClassifier
    model = XGBClassifier()
    #model=GradientBoostingClassifier()
    
    model.fit(x_train, y_train)
    
    #previsao padrao
    #y_pred = model.predict(x_test)
    
    #previsao com cutoff personalizado
    y_pred = (model.predict_proba(x_test)[:,1] >= cutoff).astype(int)
    y_probs = model.predict_proba(x_test)
    
    if config.PRINTS != 'off':
        print(' ')
        print(' ')
        print('---------[RESULTADOS DE TESTES]----------')
        print('eventos de teste:   ', len(y_test.values))
        #print('correto:      ', y_test.values)
        #print('previsto:     ', y_pred)
        #print('probabilidade:', y_probs[:,1])
        
    #plotando matriz de confusão nos dados de teste
    cf_matrix = confusion_matrix(y_test, y_pred)
    
    #sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
    #            fmt='.2%', cmap='Blues');
    
    precisao = cf_matrix[1,1]/(cf_matrix[1,1]+cf_matrix[0,1])
    odd_alvo = 1/precisao
    
    lista_precisao.append(precisao)
    lista_odd_alvo.append(odd_alvo)
    
    #verificando y_probs dos errados
    lista_erradas = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 1) & (y_test.values[i] == 0):
            lista_erradas.append(y_probs[i,1])
    
    n_entradas_previstas = y_pred.sum()
    n_entradas_erradas = len(lista_erradas)
    precisao_real = (n_entradas_previstas - n_entradas_erradas)/n_entradas_previstas
    odd_alvo_real = 1/precisao_real
    if config.PRINTS != 'off':
        print('número de entradas: ',n_entradas_previstas)
        print('entradas erradas:   ',n_entradas_erradas)
        print('precisao entradas:  ',round(precisao_real,4))
        print('odd alvo:           ',odd_alvo_real)
    
    
    #----------------------
    #previsao com cutoff personalizado
    y_valid_pred = (model.predict_proba(x_valid)[:,1] >= cutoff).astype(int) 
    y_valid_proba = model.predict_proba(x_valid)
    probabilidade = round(y_valid_proba[0][1],4)
    lista_probabilidades.append(probabilidade)
    lista_targets.append(y_valid_pred[0])
    if config.PRINTS != 'off':
        print(' ')
        print('-----------[RESPOSTA FINAL]--------------')
        print('alvo',tipo, 'que',ALVO_PRIMARIO, 'gols', '---target: 1')
        print('target:',y_valid_pred[0], '| probabilidade: ',round(y_valid_proba[0][1],4))
    
    if config.SHAPS == 'on':
        import shap
        import matplotlib.pyplot as plt
        shap_values = shap.Explainer(model).shap_values(x_test)
        f = plt.figure()
        shap.summary_plot(shap_values, x_test)


df_resultados = pd.DataFrame(lista_ensaios)
df_resultados.columns = ['test_size', 'tipo', 'alvo_primario', 'alvo_secundario']
df_resultados['target'] = lista_targets
df_resultados['probabilidade'] = lista_probabilidades
df_resultados['precisao'] = lista_precisao
df_resultados['odd_alvo'] = lista_odd_alvo
df_resultados.to_excel('df_resultados.xlsx', index=False)
   
    #https://stackoverflow.com/questions/65534163/get-a-feature-importance-from-shap-values
    #vals = np.abs(shap_values[0]).mean(0)
    #feature_names = list(x_train)
    #shap_df = pd.DataFrame(shap_values, columns=feature_names)
    #vals = np.abs(shap_df.values).mean(0)
    #shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    #shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    #first_20 = shap_importance.iloc[:20,0]
