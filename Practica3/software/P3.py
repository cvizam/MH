# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.preprocessing import MinMaxScaler
import random

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
def evaluation_function(pred,y_test,w,alpha):
    success = 0
    under_value = 0
    # Tasa de acierto
    success = np.count_nonzero(pred == y_test)
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
        
        # Preparo los datos antes de ser entrenados
        w_c = np.copy(w)
        w_c[w_c < 0.2] = 0.0
        
        # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
        neigh.fit(x_train*w_c,y_train)
        # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
        idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
        # Me quedo las etiquetas del segundo vecino
        pred = y_train[idx[:,1]]

        # Predicción de las etiquetas de los datos de test
        #prediction = neigh.predict(x_test*w)
        # Función de evaluación
        last_clas, last_red, last_agr = evaluation_function(pred,y_train,w,0.5)
        
        while n < 20*len(w) and it < 15000:
            for i in range(len(w)):
                w_copy = w.copy()
                # Valor para mutar peso
                z = np.random.normal(0,0.3,1)
                w[i] += z
                
                # Normalizo el vector de pesos
                if w[i] < 0.0:
                    w[i] = 0.0
                elif w[i] > 1.0:
                    w[i] = 1.0
                        
                # Llamo al clasificador
                neigh = KNeighborsClassifier(n_neighbors=1)
                # Preparo los datos antes de ser entrenados
                w_c = np.copy(w)
                w_c[w_c < 0.2] = 0.0
                # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
                neigh.fit(x_train*w_c,y_train)
                # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
                idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
                # Me quedo las etiquetas del segundo vecino
                pred = y_train[idx[:,1]]
                # Evalúo
                new_clas, new_red, new_agr = evaluation_function(pred,y_train,w,0.5)
                it += 1
                # Compruebo si se produce mejora
                if new_agr > last_agr:
                    last_agr = new_agr
                    break;
                else:
                    w = w_copy
                    n += 1
                     
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*w)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w,0.5)
        end = time.time()
        # Almaceno los resultados obtenidos en esa partición
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
    
    return tasa_clas, tasa_red, agregado, times

# Búsqueda Local para AM
def LS(x_train,y_train, initial_solution):
    
    # Vector de pesos
    w = np.copy(initial_solution.w)
    n = 1
    ev = 1

    # Función de evaluación
    last_agr = initial_solution.agr
        
    while n < 2*len(w):
        for i in range(len(w)):
            w_copy = w.copy()
            # Valor para mutar peso
            z = np.random.normal(0,0.3,1)
            w[i] += z
                
            # Normalizo el vector de pesos
            if w[i] < 0.0:
                w[i] = 0.0
            elif w[i] > 1.0:
                w[i] = 1.0
                        
            # Llamo al clasificador
            neigh = KNeighborsClassifier(n_neighbors=1)
            # Preparo los datos antes de ser entrenados
            wc = np.copy(w)
            wc = np.array(wc)
            wc[wc < 0.2] = 0.0
            # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
            neigh.fit(x_train*wc,y_train)
            # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
            idx = neigh.kneighbors(x_train*wc,n_neighbors=2,return_distance=False)
            # Me quedo las etiquetas del segundo vecino
            pred = y_train[idx[:,1]]
            #prediction = neigh.predict(x_test*w)
            new_clas, new_red, new_agr = evaluation_function(pred,y_train,w,0.5)
            ev += 1
            # Compruebo si se produce mejora
            if new_agr > last_agr:
                last_agr = new_agr
                break;
            else:
                w = w_copy
                n += 1
                
    return w, ev
            
###############################################################################
# Clase para representar los elementos de una poblacion
class element:
    # Constructor
    def __init__(self,weights,x_train,y_train):
        self.w = weights
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        # Preparo los datos antes de ser entrenados
        w_c = np.copy(self.w)
        w_c[w_c < 0.2] = 0.0
        # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
        neigh.fit(x_train*w_c,y_train)
        # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
        idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
        # Me quedo las etiquetas del segundo vecino
        pred = y_train[idx[:,1]]
        
        Clas, Red, Agr = evaluation_function(pred,y_train,self.w,0.5)
        self.agr = Agr
        self.clas = Clas
        self.red = Red
    
        

# Torneo Binario
def binaryTournament(population):
    # Genero 2 números aleatorios
    r1 = np.random.randint(len(population))
    r2 = np.random.randint(len(population))
    
    # Compruebo que no sean iguales, si lo son, genero otro
    while r1 == r2:
        r2 = np.random.randint(len(population))
    
    # Compruebo cual de los 2 es mejor, y lo devuelvo
    if population[r1].agr > population[r2].agr:
        return population[r1]
    else:
        return population[r2]

# Cruce BLX
def BLXCross(p1,p2,alpha,x_train,y_train):
    descendents = []
    # Genero 2 descendientes
    for i in range(2):
        h = []
        for j in range(len(p1.w)):
            Cmax = np.max([p1.w[j],p2.w[j]])
            Cmin = np.min([p1.w[j],p2.w[j]])
            I = Cmax - Cmin
            value = np.random.uniform(Cmin-I*alpha, Cmax+I*alpha)
            # Normalizo el valor
            if value > 1.0:
                value = 1.0
            elif value < 0.0:
                value = 0.0
            h.append(value)
        
        h = np.array(h)
        he = element(h,x_train,y_train)
        descendents.append(he)
    
    return descendents

# Cruce Aritmético
def arithmeticCross(p1,p2,x_train,y_train):
    descendents = []
    descendent = []
    descendent2 = []
    # Genero los descendientes
    alpha = np.random.rand(1)
    descendent = (p1.w*alpha + p2.w*(1-alpha))
    descendent2 = (p1.w*(1-alpha) + p2.w*alpha)
    
    # Normalizo los valores
    descendent[descendent > 1.0] = 1.0
    descendent[descendent < 0.0] = 0.0
    descendent2[descendent2 > 1.0] = 1.0
    descendent2[descendent2 < 0.0] = 0.0
    
    descendent = element(descendent,x_train,y_train)
    descendents.append(descendent)
    descendent2 = element(descendent2,x_train,y_train)
    descendents.append(descendent2)
    
    return descendents

# Reemplazamiento de poblaciones
def replacement(population1,population2):
    mejor = population1[0].agr
    i_m = 0
    peor = population2[0].agr
    i_p = 0
    for i in range(len(population1)):
        if population1[i].agr > mejor:
            mejor = population1[i].agr
            i_m = i
        if population2[i].agr < peor:
            peor = population2[i].agr
            i_p = i
            
    if mejor > peor:
        population2[i_p] = population1[i_m]
        
    return population2

# Mejor individuo
def bestElement(population):
    best = population[0]
    for i in range(len(population)):
        if population[i].agr > best.agr:
            best = population[i]
            
    return best

# Indice mejor individuo
def bestElementIndex(population):
    best = population[0]
    index = 0
    for i in range(len(population)):
        if population[i].agr > best.agr:
            best = population[i]
            index = i
            
    return index

# Peores individuos
def worstElements(population):
    pc = np.copy(population)
    aux = []
    for i in range(len(population)):
        aux.append((i,population[i].agr)) 

    dtype = [('index', int), ('agr', float)]
    a = np.array(aux, dtype=dtype)
    aux = np.sort(a, order='agr')
    
    population.remove(pc[aux[0][0]])
    population.remove(pc[aux[1][0]])
    
    return population
    
        
# Algoritmo Genético Generacional
def AGG(x,y,cruce):
    
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
        
        # Genero la poblacion inicial
        population = []
        for i in range(30):
            w = np.random.uniform(0,1,x.shape[1])
            ele = element(w,x_train,y_train)
            population.append(ele)
        
        ev = 30
        while ev < 15000:
            
            # Proceso de Selección de padres
            parents = []
            for i in range(len(population)):
                parents.append(binaryTournament(population))
                
            # Proceso de Cruce
            cross_numbers = np.trunc(0.7*(len(parents)/2))
            
            # Compruebo que tipo de cruce se ha elegido
            # Cruce BLX-03
            if cruce == 'BLX':
                for i in range(int(cross_numbers)):
                    des = BLXCross(parents[2*i],parents[2*i+1],0.3,x_train,y_train)
                    # Almaceno los 2 descendientes generados
                    parents[2*i] = des[0]
                    parents[2*i+1] = des[1]
                    
            # Cruce Aritmético
            else:
                for i in range(int(cross_numbers)):
                    des = arithmeticCross(parents[2*i],parents[2*i+1],x_train,y_train)
                    #Almaceno los 2 descendientes generados
                    parents[2*i] = des[0]
                    parents[2*i+1] = des[1]
            
            # Aumento las evaluaciones
            ev += cross_numbers * 2
                    
            # Proceso de Mutación
            population_genes = len(parents) * len(parents[0].w)
            
            for i in range(int(np.round(0.001 * population_genes))):
                cro = np.random.randint(len(parents))
                gen = np.random.randint(len(parents[0].w))
                z = np.random.normal(0,0.3,1)
                parents[cro].w[gen] += z
  
                # Normalizo los pesos tras la mutación
                if parents[cro].w[gen] > 1.0:
                    parents[cro].w[gen] = 1.0
                elif parents[cro].w[gen] < 0.0:
                    parents[cro].w[gen] = 0.0
                    
                # Actualizo los cromosomas mutados
                mut = parents[cro]
                mt = element(mut.w,x_train,y_train)
                parents[cro] = mt
                
            # Aumento las evaluaciones 
            ev += int(np.round(0.001 * population_genes))
                    
            # Proceso de Reemplazamiento
            parents = replacement(population,parents)
            population = np.copy(parents)
                            
                
        end = time.time()
        
        # Devuelvo el mejor de la población
        w = bestElement(population)
        
        # Evaluo la mejor 'w'
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(x_train*w.w,y_train)
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*w.w)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w.w,0.5)
        
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
               
    return tasa_clas, tasa_red, agregado, times

# Algoritmo Genético Estacionario
def AGE(x,y,cruce):
    
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
        
        # Genero la poblacion inicial
        population = []
        for i in range(30):
            w = np.random.uniform(0,1,x.shape[1])
            ele = element(w,x_train,y_train)
            population.append(ele)
        
        ev = 30
        while ev < 15000:
            
            # Proceso de Selección de padres
            parents = []
            parents.append(binaryTournament(population))
            parents.append(binaryTournament(population))
                
            # Proceso de Cruce            
            # Compruebo que tipo de cruce se ha elegido
            
            # Cruce BLX-03
            if cruce == 'BLX':
                des = BLXCross(parents[0],parents[1],0.3,x_train,y_train)
                # Almaceno los 2 descendientes generados
                parents[0] = des[0]
                parents[1] = des[1]
                    
            # Cruce Aritmético
            else:
                des = arithmeticCross(parents[0],parents[1],x_train,y_train)
                #Almaceno los 2 descendientes generados
                parents[0] = des[0]
                parents[1] = des[1]
            
            # Aumento las evaluaciones
            ev += 2
                    
            # Proceso de Mutación
            pm_cromosoma = 0.001 * len(parents[0].w)
            
            # Hijo 0
            h1 = np.random.uniform(0,1,1)
            if h1 < pm_cromosoma:
                g1 = np.random.randint(len(parents[0].w))
                z = np.random.normal(0,0.3,1)
                parents[0].w[g1] += z
                # Aumento las evaluaciones 
                ev += 1
                
                # Normalizo los pesos tras la mutación
                if parents[0].w[g1] > 1.0:
                    parents[0].w[g1] = 1.0
                elif parents[0].w[g1] < 0.0:
                    parents[0].w[g1] = 0.0
                    
                # Actualizo los cromosomas mutados
                mut = parents[0]
                mt = element(mut.w,x_train,y_train)
                parents[0] = mt
                
            # Hijo 1
            h2 = np.random.uniform(0,1,1)
            if h2 < pm_cromosoma:
                g2 = np.random.randint(len(parents[1].w))
                z = np.random.normal(0,0.3,1)
                parents[1].w[g2] += z
                # Aumento las evaluaciones 
                ev += 1
                
                # Normalizo los pesos tras la mutación
                if parents[1].w[g2] > 1.0:
                    parents[1].w[g2] = 1.0
                elif parents[1].w[g2] < 0.0:
                    parents[1].w[g2] = 0.0
                    
                # Actualizo los cromosomas mutados
                mut = parents[1]
                mt = element(mut.w,x_train,y_train)
                parents[1] = mt
                    
            
            # Proceso de Reemplazamiento
            population.append(parents[0])
            population.append(parents[1])
            population = worstElements(population)
                
        end = time.time()
        
        # Devuelvo el mejor de la población
        w = bestElement(population)
        
        # Evaluo la mejor 'w'
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(x_train*w.w,y_train)
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*w.w)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w.w,0.5)
        
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
               
    return tasa_clas, tasa_red, agregado, times
               

# Algoritmo Memético
def AM(x,y,generations,option):
    
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
        
        # Genero la poblacion inicial
        population = []
        for i in range(10):
            w = np.random.uniform(0,1,x.shape[1])
            ele = element(w,x_train,y_train)
            population.append(ele)
        
        ev = 10
        generation = 1
        while ev < 15000:
            
            # Proceso de Selección de padres
            parents = []
            for i in range(len(population)):
                parents.append(binaryTournament(population))
                
            # Proceso de Cruce
            cross_numbers = np.trunc(0.7*(len(parents)/2))
            
            # Compruebo que tipo de cruce se ha elegido
            # Cruce BLX-03
            for i in range(int(cross_numbers)):
                des = BLXCross(parents[2*i],parents[2*i+1],0.3,x_train,y_train)
                # Almaceno los 2 descendientes generados
                parents[2*i] = des[0]
                parents[2*i+1] = des[1]
            
            # Aumento las evaluaciones
            ev += cross_numbers * 2
                    
            # Proceso de Mutación
            population_genes = len(parents) * len(parents[0].w)
            num_mut = int(np.round(0.001 * population_genes))
            if num_mut == 0:
                num_mut = 1
            
            for i in range(num_mut):
                cro = np.random.randint(len(parents))
                gen = np.random.randint(len(parents[0].w))
                z = np.random.normal(0,0.3,1)
                parents[cro].w[gen] += z
  
                # Normalizo los pesos tras la mutación
                if parents[cro].w[gen] > 1.0:
                    parents[cro].w[gen] = 1.0
                elif parents[cro].w[gen] < 0.0:
                    parents[cro].w[gen] = 0.0
                    
                # Actualizo los cromosomas mutados
                mut = parents[cro]
                mt = element(mut.w,x_train,y_train)
                parents[cro] = mt
                
                
            # Aumento las evaluaciones 
            ev += num_mut
                    
            # Proceso de Reemplazamiento
            parents = replacement(population,parents)
            population = np.copy(parents)
            
            generation +=1
            
            # Compruebo que cada 10 generaciones se aplique la BL
            if generation % generations == 0:
                if option == 1:
                    for i in range(len(population)):
                        w, eva = LS(x_train,y_train,population[i])
                        wu = element(w,x_train,y_train)
                        population[i] = wu
                        ev += eva
                elif option == 2:
                    cro = np.random.randint(len(population))
                    w, eva = LS(x_train,y_train,population[cro])
                    wu = element(w,x_train,y_train)
                    population[cro] = wu
                    ev += eva
                else:
                    best = bestElementIndex(population)
                    w, eva = LS(x_train,y_train,population[best])
                    wu = element(w,x_train,y_train)
                    population[best] = wu
                    ev += eva
            
        end = time.time()
        
        # Devuelvo el mejor de la población
        w = bestElement(population)
        
        # Evaluo la mejor 'w'
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(x_train*w.w,y_train)
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*w.w)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w.w,0.5)
        
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
               
    return tasa_clas, tasa_red, agregado, times     

# Algoritmo Enfriamiento Simulado
def ES(x,y):
    
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
        
        # Generación de la solución inicial
        w = np.random.uniform(0,1,x.shape[1])
        
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        
        # Preparo los datos antes de ser entrenados
        w_c = np.copy(w)
        w_c[w_c < 0.2] = 0.0
        
        # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
        neigh.fit(x_train*w_c,y_train)
        # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
        idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
        # Me quedo las etiquetas del segundo vecino
        pred = y_train[idx[:,1]]

        # Predicción de las etiquetas de los datos de test
        #prediction = neigh.predict(x_test*w)
        # Función de evaluación
        last_clas, last_red, last_agr = evaluation_function(pred,y_train,w,0.5)
        
        # Parámetros y Ejecuciones
        best_agr = last_agr
        best_w = w
        Tf = 1e-3
        T0 = (0.3 * best_agr) / (-np.log(0.3))
        max_neighbors = 10 * len(w)
        max_successes = 0.1 * max_neighbors
        max_evaluations = 15000
        M = max_evaluations / max_neighbors
        B = (T0 - Tf) / (M * T0 * Tf)
        T = T0
        ev = 0
        n_successes = 1
        K = 1
        
        while ev < max_evaluations and n_successes > 0 and T > Tf:
            n_successes = 0
            int_ev = 0
            while int_ev < max_neighbors and n_successes < max_successes:
                int_ev += 1
                # Valor para mutar peso
                z = np.random.normal(0,0.3,1)
                caract = np.random.randint(len(w))
                w_mut = np.copy(w)
                w_mut[caract] += z
                
                
                # Normalizo el vector de pesos
                if w_mut[caract] < 0.0:
                    w_mut[caract] = 0.0
                elif w_mut[caract] > 1.0:
                    w_mut[caract] = 1.0
                    
                # Llamo al clasificador
                neigh = KNeighborsClassifier(n_neighbors=1)
                # Preparo los datos antes de ser entrenados
                w_c = np.copy(w_mut)
                w_c[w_c < 0.2] = 0.0
                # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
                neigh.fit(x_train*w_c,y_train)
                # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
                idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
                # Me quedo las etiquetas del segundo vecino
                pred = y_train[idx[:,1]]
                # Evalúo
                new_clas, new_red, new_agr = evaluation_function(pred,y_train,w_mut,0.5)
                
                # Calculo la diferencia entre las funciones objetivo
                difference = new_agr - last_agr
                
                if difference > 0 or np.random.uniform() <= np.exp(difference / (K * T)):
                    w = w_mut
                    last_agr = new_agr
                    n_successes += 1
                    
                    if last_agr > best_agr:
                        best_agr = last_agr
                        best_w = w
                
            ev += int_ev
            T = T / (1 + B * T)
                
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        # Preparo los datos antes de ser entrenados  
        neigh.fit(x_train*best_w,y_train)             
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*best_w)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,best_w,0.5)
        end = time.time()
        # Almaceno los resultados obtenidos en esa partición
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
    
    return tasa_clas, tasa_red, agregado, times

# Búsqueda Local para ILS
def LS_ILS(x_train,y_train, initial_solution):
    
    # Vector de pesos
    w = np.copy(initial_solution)
    
    # Llamo al clasificador
    neigh = KNeighborsClassifier(n_neighbors=1)
        
    # Preparo los datos antes de ser entrenados
    w_c = np.copy(w)
    w_c[w_c < 0.2] = 0.0
    
    # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
    neigh.fit(x_train*w_c,y_train)
    # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
    idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
    # Me quedo las etiquetas del segundo vecino
    pred = y_train[idx[:,1]]
    
    # Predicción de las etiquetas de los datos de test
    # Función de evaluación
    last_clas, last_red, last_agr = evaluation_function(pred,y_train,w,0.5)
    
    ev = 1

    while ev < 1000:
        for i in range(len(w)):
            w_copy = w.copy()
            # Valor para mutar peso
            z = np.random.normal(0,0.3,1)
            w[i] += z
                
            # Normalizo el vector de pesos
            if w[i] < 0.0:
                w[i] = 0.0
            elif w[i] > 1.0:
                w[i] = 1.0
                        
            # Llamo al clasificador
            neigh = KNeighborsClassifier(n_neighbors=1)
            # Preparo los datos antes de ser entrenados
            wc = np.copy(w)
            wc = np.array(wc)
            wc[wc < 0.2] = 0.0
            # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
            neigh.fit(x_train*wc,y_train)
            # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
            idx = neigh.kneighbors(x_train*wc,n_neighbors=2,return_distance=False)
            # Me quedo las etiquetas del segundo vecino
            pred = y_train[idx[:,1]]
            #prediction = neigh.predict(x_test*w)
            new_clas, new_red, new_agr = evaluation_function(pred,y_train,w,0.5)
            ev += 1
            # Compruebo si se produce mejora
            if new_agr > last_agr:
                last_agr = new_agr
                break;
            else:
                w = w_copy
                
    return w             
                
# Algoritmo Búsqueda Local Reiterada
def ILS(x,y):
    
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
        
        # Generación de la solución inicial
        w = np.random.uniform(0,1,x.shape[1])
        
        w = LS_ILS(x_train,y_train,w)
        
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        
        # Preparo los datos antes de ser entrenados
        w_c = np.copy(w)
        w_c[w_c < 0.2] = 0.0
        
        # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
        neigh.fit(x_train*w_c,y_train)
        # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
        idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
        # Me quedo las etiquetas del segundo vecino
        pred = y_train[idx[:,1]]

        # Función de evaluación
        last_clas, last_red, last_agr = evaluation_function(pred,y_train,w,0.5)
        
        max_iter = 15
        it = 1
        t = int(0.1 * len(w))
        best_agr = last_agr
        
        while it < max_iter:
            it += 1
            
            # Mutación
            w_mut = np.copy(w)
            n = np.arange(len(w))
            n = set(n)
            index = random.sample(n,t)
            # Valor para mutar peso
            z = np.random.normal(0,0.4,1)
            w_mut[index] += z
            
            # Normalizo el vector de pesos
            for i in index:
                if w_mut[i] < 0.0:
                    w_mut[i] = 0.0
                elif w_mut[i] > 1.0:
                    w_mut[i] = 1.0
            
            # Llamo a Búsqueda Local
            
            w_mut = LS_ILS(x_train,y_train,w_mut)
                    
            # Llamo al clasificador
            neigh = KNeighborsClassifier(n_neighbors=1)
            # Preparo los datos antes de ser entrenados
            w_c = np.copy(w_mut)
            w_c[w_c < 0.2] = 0.0
            # Entrenamiento a partir de datos de entrenamiento y sus etiquetas
            neigh.fit(x_train*w_c,y_train)
            # Genero 2 vecinos: me quedo con el segundo para cumplir "leave-one-out"
            idx = neigh.kneighbors(x_train*w_c,n_neighbors=2,return_distance=False)
            # Me quedo las etiquetas del segundo vecino
            pred = y_train[idx[:,1]]
            # Evalúo
            new_clas, new_red, new_agr = evaluation_function(pred,y_train,w_mut,0.5)
            
            if new_agr > best_agr:
                best_agr = new_agr
                w = w_mut
                
                
         # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(x_train*w,y_train)
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*w)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w,0.5)
        end = time.time()
        # Almaceno los resultados obtenidos en esa partición
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
    
    return tasa_clas, tasa_red, agregado, times       


# Modelo para ED
def rand_1(population,index,CR,F,x_train,y_train):
    offspring = []
    r1 = np.random.randint(len(population))
    # Compruebo que no sea igual al índice
    while r1 == index:
        r1 = np.random.randint(len(population))
    r2 = np.random.randint(len(population))
    # Compruebo que no sea igual al índice ó r1
    while r2 == index or r2 == r1:
        r2 = np.random.randint(len(population))
    r3 = np.random.randint(len(population))
    # Compruebo que no sea igual al índice ó r1 ó r2
    while r3 == index or r3 == r1 or r3 == r2:
        r3 = np.random.randint(len(population))
      
    for k in range(len(population[0].w)):
        if np.random.uniform(0,1,1) <= CR:
            offspring.append(population[r1].w[k] + F * (population[r2].w[k] - population[r3].w[k]))
        else:
            offspring.append(population[index].w[k])
    
    offspring = np.array(offspring)
    
    # Normalizo los valores
    offspring[offspring > 1.0] = 1.0
    offspring[offspring < 0.0] = 0.0
            
    offspring = element(offspring,x_train,y_train)
    
    return offspring

# Modelo para ED
def current_to_best_1(population,index,CR,F,x_train,y_train):
    offspring = []
    r1 = np.random.randint(len(population))
    # Compruebo que no sea igual al índice
    while r1 == index:
        r1 = np.random.randint(len(population))
    r2 = np.random.randint(len(population))
    # Compruebo que no sea igual al índice ó r1
    while r2 == index or r2 == r1:
        r2 = np.random.randint(len(population))
    
    best = bestElement(population)
    
    for k in range(len(population[0].w)):
        if np.random.uniform(0,1,1) <= CR:
            offspring.append(population[index].w[k] + F * (best.w[k] - population[index].w[k])
                                                    + F * (population[r1].w[k] - population[r2].w[k]))
        else:
            offspring.append(population[index].w[k])
    
    offspring = np.array(offspring)
    
    # Normalizo los valores
    offspring[offspring > 1.0] = 1.0
    offspring[offspring < 0.0] = 0.0
            
    offspring = element(offspring,x_train,y_train)
    
    return offspring
    
# Algoritmo Evolución Diferencial   
def ED(x,y,version):
    
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
        
        # Genero la poblacion inicial
        population = []
        for i in range(50):
            w = np.random.uniform(0,1,x.shape[1])
            ele = element(w,x_train,y_train)
            population.append(ele)
        
        ev = 50
        CR = 0.5
        F = 0.5
        
        while ev < 15000:
            
            for i in range(len(population)):
                
                if version == 'RAND_1':
                    offspring = rand_1(population,i,CR,F,x_train,y_train)
                    ev += 1
                elif version == 'CURRENT_TO_BEST_1':
                    offspring = current_to_best_1(population,i,CR,F,x_train,y_train)
                    ev += 1
                
                if offspring.agr > population[i].agr:
                    population[i] = offspring
                
        end = time.time()
        
        # Devuelvo el mejor de la población
        w = bestElement(population)
        
        # Evaluo la mejor 'w'
        # Llamo al clasificador
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(x_train*w.w,y_train)
        # Predicción de las etiquetas de los datos de test
        prediction = neigh.predict(x_test*w.w)
        # Función de evaluación
        clas, red, agr = evaluation_function(y_test,prediction,w.w,0.5)
        
        tasa_clas.append(clas)
        tasa_red.append(red)
        agregado.append(agr)
        times.append(end-start)
               
    return tasa_clas, tasa_red, agregado, times
            
###############################################################################
# EJECUCIONES

print('--- Colposcopy Dataset ---\n')
x, y = load('datos/colposcopy.csv')
x = MinMaxScaler().fit_transform(x)

# EJECUCIÓN ALGORITMO ENFRIAMIENTO SIMULADO

tasa_clas, tasa_red, agregado, times = ES(x,y)
display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO BL REITERADA

#tasa_clas, tasa_red, agregado, times = ILS(x,y)
#display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO EVOLUCIÓN DIFERENCIAL RAND/1

#tasa_clas, tasa_red, agregado, times = ED(x,y,'RAND_1')
#display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO EVOLUCIÓN DIFERENCIAL CURRENT_TO_BEST/1

#tasa_clas, tasa_red, agregado, times = ED(x,y,'CURRENT_TO_BEST_1')
#display_table(tasa_clas,tasa_red,agregado,times)

###############################################################################

print('--- Ionosphere Dataset ---\n')
x, y = load('datos/ionosphere.csv')
x = MinMaxScaler().fit_transform(x)

# EJECUCIÓN ALGORITMO ENFRIAMIENTO SIMULADO

tasa_clas, tasa_red, agregado, times = ES(x,y)
display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO BL REITERADA

#tasa_clas, tasa_red, agregado, times = ILS(x,y)
#display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO EVOLUCIÓN DIFERENCIAL RAND/1

#tasa_clas, tasa_red, agregado, times = ED(x,y,'RAND_1')
#display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO EVOLUCIÓN DIFERENCIAL CURRENT_TO_BEST/1

#tasa_clas, tasa_red, agregado, times = ED(x,y,'CURRENT_TO_BEST_1')
#display_table(tasa_clas,tasa_red,agregado,times)

###############################################################################

print('\n--- Texture Dataset ---\n')
x, y = load('datos/texture.csv')
x = MinMaxScaler().fit_transform(x)

# EJECUCIÓN ALGORITMO ENFRIAMIENTO SIMULADO

tasa_clas, tasa_red, agregado, times = ES(x,y)
display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO BL REITERADA

#tasa_clas, tasa_red, agregado, times = ILS(x,y)
#display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO EVOLUCIÓN DIFERENCIAL RAND/1

#tasa_clas, tasa_red, agregado, times = ED(x,y,'RAND_1')
#display_table(tasa_clas,tasa_red,agregado,times)


# EJECUCIÓN ALGORITMO EVOLUCIÓN DIFERENCIAL CURRENT_TO_BEST/1

#tasa_clas, tasa_red, agregado, times = ED(x,y,'CURRENT_TO_BEST_1')
#display_table(tasa_clas,tasa_red,agregado,times)
