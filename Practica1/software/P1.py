# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1) # Fijo la semilla


###############################################################################
# FUNCIONES

# Carga y divide los datos y sus etiquetas
def load(rute):
    data = pd.read_csv(rute,header=-1)
    data = np.array(data)

    x = data[:,:-1]
    y = data[:,-1]
  
    return np.array(x), np.array(y)

# Calcula la distancia euclidea entre dos ejemplos
def euclidean_distance(x1,x2):
    distance = 0.0
    distance  = np.sum((x1 - x2)**2)
    return distance

# Calcula la media de los valores obtenidos en cada partición
def mean_calculus(values):
    summ = np.sum(values)
    return summ/5.0

# Ejemplo más cercano con misma clase y clase distinta (amigo,enemigo)
def closest_examples(x_train,e,y_train):
    closest_enemy = 0
    closest_friend = 0
    element = x_train[e]
    e_class = y_train[e]
    enemy_dist = np.inf
    friend_dist = np.inf
    
    for i in range(len(x_train)):
        if i != e:
            dist = euclidean_distance(element,x_train[i])
            # Posicion del enemigo más cercano
            if dist < enemy_dist and e_class != y_train[i]:
                enemy_dist = dist
                closest_enemy = i
            # Posición del amigo más cercano
            if dist < friend_dist and e_class == y_train[i]:
                friend_dist = dist
                closest_friend = i
    
    return x_train[closest_enemy], x_train[closest_friend]

# Función de evaluación: calcula la tasa_clas, la tasa_red y la agregación
# de una partición
def evaluation_function(y_test,prediction,w,alpha):
    success = 0
    under_value = 0
    # Tasa de acierto
    success = np.count_nonzero(prediction == y_test)
    tasa_clas = (100 * (success / len(y_test)))
    
    # Tasa de reducción
    under_value = np.count_nonzero(w<0.2)
    tasa_red = (100 * (under_value / len(w)))
    
    # Agregación
    agr = alpha * tasa_clas + (1-alpha) * tasa_red
    
    return tasa_clas, tasa_red, agr

# Representa los resultados obtenidos en cada partición y la media de ellos  
def display_table(tasa_clas,tasa_red,agregado,times):
    print('- Particion 1 -') 
    print('%clas: ',tasa_clas[0],'%red: ',tasa_red[0],'agr: ',agregado[0],'T: ',times[0],'\n')
    print('- Particion 2 -') 
    print('%clas: ',tasa_clas[1],'%red: ',tasa_red[1],'agr: ',agregado[1],'T: ',times[1],'\n')
    print('- Particion 3 -') 
    print('%clas: ',tasa_clas[2],'%red: ',tasa_red[2],'agr: ',agregado[2],'T: ',times[2],'\n')
    print('- Particion 4 -') 
    print('%clas: ',tasa_clas[3],'%red: ',tasa_red[3],'agr: ',agregado[3],'T: ',times[3],'\n')
    print('- Particion 5 -') 
    print('%clas: ',tasa_clas[4],'%red: ',tasa_red[4],'agr: ',agregado[4],'T: ',times[4],'\n')
    print('- Media -')
    print('%class: ',mean_calculus(tasa_clas),'%red: ',mean_calculus(tasa_red),'agr: ',mean_calculus(agregado),'T: ',mean_calculus(times),'\n')
    
###############################################################################
# ALGORITMOS
    
# Clasificador 1-NN
def KNN(x,y):
    
    tasa_clas = []
    tasa_red = []
    times = []
    agregado = []
   
    # Divido el conjunto de datos en 5 particiones al 80-20 %
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # Itero las 5 particiones
    for train_index, test_index in skf.split(x,y):
        # Obtengo el tiempo de inicio
        start = time.time() 
        # Obtengo los conjunto de training y de test
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        
        # Genero el vector de pesos inicializado a 1
        w = np.ones(x.shape[1])
        
        # Clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
        neigh.fit(x_train,y_train)
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w,0.5)
        # Obtengo el tiempo final
        end = time.time() # Obtengo el tiempo de fin
        
        # Almaceno los resultados obtenidos en esa partición
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
    
    return tasa_clas, tasa_red, agregado, times


# Solución Greedy, método RELIEF
def relief(x,y):
    
    tasa_clas = []
    tasa_red = []
    agregado = []
    times = []
   
    # Divido el conjunto de datos en 5 particiones al 80-20 %
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1)
    # Itero las 5 particiones
    for train_index, test_index in skf.split(x,y):
         # Obtengo el tiempo de inicio
        start = time.time()
        # Obtengo los conjunto de training y test
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        
        # Genero el vector de pesos inicializado a 0
        w = np.zeros(x.shape[1])
        
        for i in range(len(x_train)):
            # Busco el enemigo, amigo más cercano para cada ejemplo
            closest_e, closest_f = closest_examples(x_train,i,y_train)
            # Actualizo el vector de pesos
            w = w + np.abs(x_train[i] - closest_e) - np.abs(x_train[i] - closest_f)
        
        # Normalizo el vector de pesos
        wm = np.max(w)
        for i in range(len(w)):
            if w[i] < 0.0:
                w[i] = 0.0
            else:
                w[i] = w[i] / wm
        
        # Multiplico los datos por el vector de pesos
        x_train = x_train * w
        x_test = x_test * w

        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
        neigh.fit(x_train,y_train)
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w,0.5)
        # Obtengo el tiempo final
        end = time.time() # Obtengo el tiempo de fin
        
        # Almaceno los resultados obtenidos en esa partición
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
    
    return tasa_clas, tasa_red, agregado, times

# Búsqueda Local
def local_search(x,y):
    
    tasa_clas = []
    tasa_red = []
    times = []
    agregado = []
    
    # Divido el conjunto de datos en 5 particiones al 80-20 %
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1)
    # Itero las 5 particiones
    for train_index, test_index in skf.split(x,y):
        # Obtengo el tiempo de inicio
        start = time.time() 
        # Obtengo los conjunto de training y test
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        
        # Vector de pesos
        w = np.random.uniform(0,1,x.shape[1])
        n = 1
        it = 1
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
        neigh.fit(x_train*w,y_train)
        # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
        idx = neigh.kneighbors(x_test*w,n_neighbors=2,return_distance=False)
        # Me quedo las etiquetas del segundo vecino
        y_test = y_train[idx[:,1]]
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*w)
        # Función de evaluación
        last_clas, last_red, last_agr = evaluation_function(y_test,prediction,w,0.5)
        
        while n < 20*len(w) and it < 15000:
            for i in range(len(w)):
                w_copy = w.copy()
                # Valor para mutar peso
                z = np.random.normal(0,0.3,1)
                w[i] += z
                
                # Normalizo el vector de pesos
                if w[i] < 0.2:
                    w[i] = 0.0
                elif w[i] > 1.0:
                    w[i] = 1.0
                        
                # Llamo al clasificador
                neigh = KNeighborsClassifier(n_neighbors=1)
                neigh.fit(x_train*w,y_train)
                idx = neigh.kneighbors(x_test*w,n_neighbors=2,return_distance=False)
                
                y_test = y_train[idx[:,1]]
                prediction = neigh.predict(x_test*w)
                new_clas, new_red, new_agr = evaluation_function(y_test,prediction,w,0.5)
                it += 1
                # Compruebo si se produce mejora
                if new_agr > last_agr:
                    last_agr = new_agr
                    break;
                else:
                    w = w_copy
                    n += 1
                    
        end = time.time()
        final_clas = new_clas
        final_red = new_red
        final_agr = new_agr
           
        # Almaceno los resultados obtenidos en esa partición         
        tasa_clas.append(final_clas)
        tasa_red.append(final_red)
        agregado.append(final_agr)
        times.append(end-start)
    
    return tasa_clas, tasa_red, agregado, times
                
            
###############################################################################
# EJECUCIONES
    
print('--- Colposcopy Dataset ---\n')
x, y = load('datos/colposcopy.csv')
x = MinMaxScaler().fit_transform(x)

print(' * 1NN * \n')
tasa_clas, tasa_red, agregado, times = KNN(x,y)
display_table(tasa_clas,tasa_red,agregado,times)

print('\n * Relief * \n')
tasa_clas, tasa_red, agregado, times = relief(x,y)
display_table(tasa_clas,tasa_red,agregado,times)

print('\n * Local Search * \n')
tasa_clas, tasa_red, agregado, times = local_search(x,y)
display_table(tasa_clas,tasa_red,agregado,times)


###############################################################################

print('--- Ionosphere Dataset ---\n')
x, y = load('datos/ionosphere.csv')
x = MinMaxScaler().fit_transform(x)

print(' * 1NN * \n')
tasa_clas, tasa_red, agregado, times = KNN(x,y)
display_table(tasa_clas,tasa_red,agregado,times)

print('\n * Relief * \n')
tasa_clas, tasa_red, agregado, times = relief(x,y)
display_table(tasa_clas,tasa_red,agregado,times)

print('\n * Local Search * \n')
tasa_clas, tasa_red, agregado, times = local_search(x,y)
display_table(tasa_clas,tasa_red,agregado,times)

###############################################################################

print('\n--- Texture Dataset ---\n')
x, y = load('datos/texture.csv')
x = MinMaxScaler().fit_transform(x)

print(' * 1NN * \n')
tasa_clas, tasa_red, agregado, times = KNN(x,y)
display_table(tasa_clas,tasa_red,agregado,times)

print('\n * Relief * \n')
tasa_clas, tasa_red, agregado, times = relief(x,y)
display_table(tasa_clas,tasa_red,agregado,times)

print('\n * Local Search * \n')
tasa_clas, tasa_red, agregado, times = local_search(x,y)
display_table(tasa_clas,tasa_red,agregado,times)


        





            
    



