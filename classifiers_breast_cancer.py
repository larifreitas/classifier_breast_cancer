
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def svm():
    print("-------------------------CLASSIFICADOR SVM:")
    # padrão de kernel rbf
    svmClass = SVC()

    # validação cruzada com 7 conjunts diferentes para generalização
    scoresSVM = cross_validate(svmClass, dados, classes, cv=5, scoring=mets)

    # validação cruzada
    for s in scoresSVM:
        print("%s || Média: %.3f || Desvio Padrão: %.3f" 
              % (s, 
                 np.average(scoresSVM[s]), 
                 np.std(scoresSVM[s])))
        
    #previsão
    predCross = cross_val_predict(svmClass, dados, classes, cv=5)

    # Matriz de confusão
    confMatrix = confusion_matrix(classes, predCross)
    print("\nMatriz de confusão:\n",confMatrix, "\n\n")

    # demonstração em imagem
    demo_confiMatrix(confMatrix, 'SVM')

def knn():
    print("-------------------------CLASSIFICADOR KNN:")
    knn = KNeighborsClassifier()
    scoresKNN = cross_validate(knn,dados,classes,cv=5,scoring=mets)

    #validação cruzada
    for s in scoresKNN:
        print("%s || Média: %.3f || Desvio Padrão: %.3f" 
                % (s, 
                    np.average(scoresKNN[s]), 
                    np.std(scoresKNN[s])))
        
    # previsão
    pred = cross_val_predict(knn,dados,classes,cv=5)

    # Matriz de confusão
    confMatrix = confusion_matrix(classes,pred)
    print("\nMatriz de confusão:\n",confMatrix, "\n\n")

    # demonstração em imagem
    demo_confiMatrix(confMatrix, 'KNN')

def NB():
    print("-------------------------CLASSIFICADOR NAIVE BAYES:")
    nb = GaussianNB()
    scoresKNN = cross_validate(nb,dados,classes,cv=5,scoring=mets)

    for s in scoresKNN:
        print("%s || Média: %.3f || Desvio Padrão: %.3f" 
                % (s, 
                    np.average(scoresKNN[s]), 
                    np.std(scoresKNN[s])))
        
    pred = cross_val_predict(nb,dados,classes,cv=5)

    confMatrix = confusion_matrix(classes,pred)
    print("\nMatriz de confusão:\n",confMatrix, "\n\n")

    demo_confiMatrix(confMatrix, 'Naive Bayes')



def demo_confiMatrix(conf_matrix, classifier):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title(f'Matriz de confusão - {classifier}')
    plt.colorbar()
    plt.xticks(np.arange(2), ['MALIGNO', 'BENÍGNO'])
    plt.yticks(np.arange(2), ['MALIGNO', 'BENÍGNO'])
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_{classifier}.png')



if __name__ == "__main__":
    # dataset
    cancer_data = pd.read_csv('/home/larissa/Unimar/FundamentosDeIA/Atividade/breast-cancer.csv')

    # ler/manipular linhas e colunas dataset
    cancer_data['diagnosis'] = cancer_data['diagnosis'].astype('category')
    cancer_data.values[:,2:] = cancer_data.values[:,2:].astype(np.float64)
    cancer_data = cancer_data.drop(columns=['id'])
    dados = cancer_data['diagnosis'].astype('category')
    dados.drop(columns=['diagnosis']).values

    dados = cancer_data.drop(columns=['diagnosis']).values
    classes = cancer_data['diagnosis'].values

    # métricas
    mets = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

    # selecionar classificadores
    svm()
    knn()
    NB()
