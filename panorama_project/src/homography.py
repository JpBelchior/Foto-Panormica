import numpy as np

#  1. NORMALIZAÇÃO DE HARTLEY

def normalizar_pontos(pts):
    """
    Normalizamos aqui os pontos para que os valores fiquem menores e deixar os proximos calculos mais tranquilos.
    Escolhemos um centroide e o normalizamos.
    Retornamos os pontos e a Transformação feita para depois desfazermos.
    """

    pts = np.array(pts, dtype=np.float64)

    # Centroide
    cx, cy = np.mean(pts, axis=0)

    # Distância média de cada ponto ao centroide
    distancias = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    dist_media = np.mean(distancias)

    # Escala para que a distância média vire √2
    s = np.sqrt(2) / dist_media

    # Matriz de normalização
    T = np.array([
        [s,  0, -s * cx],
        [0,  s, -s * cy],
        [0,  0,       1]
    ])

    # Converte pontos para coordenadas homogêneas 
    pts_h = np.column_stack([pts, np.ones(len(pts))])

    # Aplica T em cada ponto
    pts_norm = (T @ pts_h.T).T

    return pts_norm[:, :2], T


#  2. MONTAR A MATRIZ A

def montar_matriz_A(pts1, pts2):
    """
    Para cada par (x,y) → (x',y'), elimina lambda dividindo pela terceira linha.
    Isso gera 2 equações por par, todas com lado direito zero: Ah = 0

    1) h1x1+ h2y1 + h3 = lambda1
    2) h4x1+ h5y1 + h6 = lambda1
    3) h7x1+ h8y1 + h9 = lambda1

    dividindo 1 e 2 por 3

    linha 1: [-x, -y, -1,  0,  0,  0,  x·x',  y·x',  x']
    linha 2: [  0,  0,  0, -x, -y, -1,  x·y',  y·y',  y']

    Com N pares: 2N equações, 9 incógnitas (h1...h9).
    """

    A = []

    for (x, y), (xl, yl) in zip(pts1, pts2):
        A.append([-x, -y, -1,  0,   0,  0,  x*xl,  y*xl,  xl])
        A.append([ 0,  0,  0, -x,  -y, -1,  x*yl,  y*yl,  yl])

    return np.array(A, dtype=np.float64)


#  3. RESOLVER H VIA EQUAÇÕES NORMAIS

def resolver_homografia(A):
    """
    Resolve Ah = 0 (dada a montagem feita aciam)  com ||h|| = 1 via equações normais.

    Queremos minimizar ||Ah||² com ||h|| = 1.

    ||Ah||² = (Ah)t(Ah) = ht AtA h

    Então o problema é:
    minimizar  hᵀ AᵀA h = ||A.h|| isso é igual a zero para 4 pontos, porém quando vamos pra mais pontos queremos o menor valor, como se fosse a reta que se aproxima mais de 3 ponto nao colineares
    hᵀh = 1 escolhemos isso pois a geometria projetiva é definida a menos de u ma escala

    seja ||A.h|| = lambda

    ht.AtAh= lambda (multiplique por h)
    I.At,A.h = lambda.h

    Isso é um problema de autovetor.
    AtA · h = lambda · h

    Onde lambda é o menor autovalor de AtA, pois quanto menor lambda, menor é ht AtA h = lambda · hth = lambda.

    O h que queremos é o autovetor de M associado ao menor lambda.
    """

    M = A.T @ A                          

    #  M·h = lambda·h para matrizes simétricas M é sempre simetrica 
    # função eight retorna autovalores em ordem crescente 
    autovalores, autovetores = np.linalg.eigh(M)

    h = autovetores[:, 0] # autovetor do menor lambda

    H = h.reshape(3, 3)

    return H

#  4. DESNORMALIZAR H

def desnormalizar_H(H_norm, T1, T2):
    """
    Desfazemos a normalização feita na função 1, puxando a transformação lá utilizada.
    """

    T2_inv = np.linalg.inv(T2)

    H = T2_inv @ H_norm @ T1

    # Normaliza para que H[2,2] = 1
    H = H / H[2, 2]

    return H


#  5. FUNÇÃO PRINCIPAL

def calcular_homografia(pts1, pts2):
    """
    Junta tudo num passo a passo:

    Normaliza → monta A (sistema Ah=0) → forma AᵀA → acha autovetor
    do menor autovalor → desnormaliza
    """

    assert len(pts1) == len(pts2), "Número de pontos deve ser igual"
    assert len(pts1) >= 4, "Mínimo de 4 pares de pontos"

    # Passo 1: normaliza
    pts1_norm, T1 = normalizar_pontos(pts1)
    pts2_norm, T2 = normalizar_pontos(pts2)

    # Passo 2: monta A
    A = montar_matriz_A(pts1_norm, pts2_norm)

    print(f"  A: {A.shape[0]} equações, {A.shape[1]} incógnitas")

    # Passo 3: equações normais autovetor de AᵀA
    H_norm = resolver_homografia(A)

    # Passo 4: desnormaliza
    H = desnormalizar_H(H_norm, T1, T2)

    return H


#  6. ERRO DE REPROJEÇÃO

def erro_reprojecao(H, pts1, pts2):
    """
    Tentativa de ver o quao proximo nossas escolhas ficaram do perfeito
    """

    erros = []

    for pt1, pt2 in zip(pts1, pts2):

        # Coordenada homogênea de pt1
        pt1_h = np.array([pt1[0], pt1[1], 1.0])

        # Projeta com H
        pt2_proj = H @ pt1_h

        # Divide por w para voltar a coordenadas 2D
        pt2_proj = pt2_proj / pt2_proj[2]

        # Distância até o pt2 real
        erro = np.sqrt((pt2_proj[0] - pt2[0])**2 + (pt2_proj[1] - pt2[1])**2)
        erros.append(erro)

    erros = np.array(erros)

    return erros, np.mean(erros)