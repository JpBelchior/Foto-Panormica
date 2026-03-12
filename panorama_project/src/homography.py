import numpy as np



#  1. NORMALIZAÇÃO DE HARTLEY

def normalizar_pontos(pts):
    """
   Normalizamos aqui os pontos para que os valores fiquem menores e deixar os proximos calculos mais tranquilos
   escolhemos um centroide e o normalizamos. 
   Retornamos os pontos e a Transformação feita para depois desfazermos
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

    # Converte pontos para coordenadas homogêneas [x, y, 1]
    pts_h = np.column_stack([pts, np.ones(len(pts))])

    # Aplica T em cada ponto
    pts_norm = (T @ pts_h.T).T

    return pts_norm[:, :2], T



#  2. MONTAR A MATRIZ A

def montar_matriz_A(pts1, pts2):
    """
    Vamos fazer a seguinte transformção (xi,yi) <-> (xi',yi'), H @ [x, y, 1] = [x', y', 1]
    gera 2 linhas na matriz A.

    O sistema é Ah = 0, onde h = [h1,h2,...,h9]

    Esse seria o nosso conjunto de equaçoes:
    h1*x + h2*y + h3*1 = x' 
    h4*x + h5*y + h6*1 = y' 
    h7*x + h8*y + h9*1 = 1

    apos contas... 
    h1*x + h2*y + h3 - x'*h7*x - x'*h8*y - x'*h9 = 0
    h4*x + h5*y + h6 - y'*h7*x - y'*h8*y - y'*h9 = 0

    linha 1:  [ x,  y,  1,  0,  0,  0,  -x'*x,  -x'*y,  -x']
    linha 2:  [ 0,  0,  0,  x,  y,  1,  -y'*x,  -y'*y,  -y'] isso para cada ponto 

    """

    A = []

    for (x, y), (xl, yl) in zip(pts1, pts2):
        # Linha 1 do par
        A.append([-x, -y, -1,   0,  0,  0,  x*xl,  y*xl,  xl])
        # Linha 2 do par
        A.append([ 0,  0,  0,  -x, -y, -1,  x*yl,  y*yl,  yl])

    return np.array(A, dtype=np.float64)



#  3. RESOLVER H VIA SVD

def resolver_homografia_svd(A):
    """
    Resolve Ah = 0 pela SVD de A.

    A = U @ Σ @ Vt
    A solução é a última coluna de V (= última linha de Vt)
    que corresponde ao menor valor singular.

    Por que o último vetor de V?
    - Ele é o vetor que minimiza ||Ah|| com ||h|| = 1
    - Ou seja: a "direção" que A mais "esmaga" a junção, sentimos menos ao visualizar
    """

    U, S, Vt = np.linalg.svd(A)

    # Último vetor linha de Vt = último vetor coluna de V
    h = Vt[-1]

    # Reshape de (9,) para (3,3)
    H = h.reshape(3, 3)

    return H


#  4. DESNORMALIZAR H

def desnormalizar_H(H_norm, T1, T2):
    """
  Basicamente aqui desfazemos a normalização feita na função 1 , puxando a Tranfromação la utilizada
    """

    T1_inv = np.linalg.inv(T1)
    T2_inv = np.linalg.inv(T2)

    H = T2_inv @ H_norm @ T1

    # Normaliza para que H[2,2] = 1 
    H = H / H[2, 2]

    return H




def calcular_homografia(pts1, pts2):
    """
  Juntamos tudo num passo a passo

  Normalizamos-> Montamos a matriz A dados os pontos normalizados através do sistema e equações->
  Resolvemos a homografiafazendo rotaçoes e alongamento de veotres achando o valor que gera uma menor mudança-> Desnormaliza
    """

    assert len(pts1) == len(pts2), "Número de pontos deve ser igual"
    assert len(pts1) >= 4, "Mínimo de 4 pares de pontos"

    # Passo 1: normaliza
    pts1_norm, T1 = normalizar_pontos(pts1)
    pts2_norm, T2 = normalizar_pontos(pts2)

    # Passo 2: monta A
    A = montar_matriz_A(pts1_norm, pts2_norm)

    # Passo 3: SVD → H no espaço normalizado
    H_norm = resolver_homografia_svd(A)

    # Passo 4: desnormaliza
    H = desnormalizar_H(H_norm, T1, T2)

    return H


# ─────────────────────────────────────────────
#  6. ERRO DE REPROJEÇÃO
# ─────────────────────────────────────────────

def erro_reprojecao(H, pts1, pts2):
    """
    Mede o quão boa é a homografia H.
    bem simples, nos mostra a distinção dospontos, projeta pt1 com H e mede distância até pt2.
    
    """

    erros = []

    for pt1, pt2 in zip(pts1, pts2):

        # Coordenada homogênea de pt1
        pt1_h = np.array([pt1[0], pt1[1], 1.0])

        # Projeta com H
        pt2_proj = H @ pt1_h

        # Divide por w para voltar a coordenadas 2D
        pt2_proj = pt2_proj / pt2_proj[2]

        # Distância euclidiana até o pt2 real
        erro = np.sqrt((pt2_proj[0] - pt2[0])**2 + (pt2_proj[1] - pt2[1])**2)
        erros.append(erro)

    erros = np.array(erros)

    return erros, np.mean(erros)


# ─────────────────────────────────────────────
#  TESTE RÁPIDO (rode esse arquivo diretamente)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Pontos fictícios para testar (imagem 800x600)
    # Simula uma translação de 50px em x e 20px em y
    pts1 = [[180, 100], [700, 100], [700, 500], [100, 500]]
    pts2 = [[150, 120], [750, 120], [750, 520], [150, 520]]

    H = calcular_homografia(pts1, pts2)

    print("Homografia H:")
    print(np.round(H, 4))

    erros, erro_medio = erro_reprojecao(H, pts1, pts2)

    print(f"\nErro por ponto (px): {np.round(erros, 4)}")
    print(f"Erro médio: {erro_medio:.4f} px")

    # Mostra os valores singulares de A (para curiosidade)
    pts1_norm, T1 = normalizar_pontos(pts1)
    pts2_norm, T2 = normalizar_pontos(pts2)
    A = montar_matriz_A(pts1_norm, pts2_norm)
    _, S, _ = np.linalg.svd(A)
    print(f"\nValores singulares de A:")
    print(np.round(S, 4))
    print(f"Razão S[-2]/S[-1] (quanto maior, melhor): {S[-2]/S[-1]:.2f}")