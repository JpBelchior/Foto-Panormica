import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
#  POR QUÊ ESSE MÓDULO EXISTE?
#
#  A homografia H nos diz *como* transformar pontos da img2 para o espaço
#  da img1. O warper usa essa informação para criar o panorama de fato:
#  define um canvas grande o suficiente, distorce a img2, e combina as duas.
# ─────────────────────────────────────────────────────────────────────────────


def calcular_canvas(H, img1, img2):
    """
    Descobre o tamanho do canvas panorâmico e o offset necessário.

    A ideia:
      - img1 já está no lugar certo (coordenadas (0,0) até (w1,h1))
      - Os 4 cantos da img2 são projetados via H para o espaço da img1
      - O canvas engloba TODOS esses pontos (cantos da img1 + cantos projetados da img2)

    Se algum ponto projetado tiver coordenada negativa (img2 "vaza" para a
    esquerda ou para cima), aplicamos um offset para transladar tudo de volta
    ao quadrante positivo.

    Retorna:
      canvas_size : (largura, altura) do canvas em pixels
      offset      : (ox, oy) — deslocamento a aplicar em todos os pontos
    """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Cantos da img1 em coordenadas homogêneas 
    # Ordem: topo-esquerdo, topo-direito, baixo-direito, baixo-esquerdo
    cantos_img1 = np.array([
        [0,  0,  1],
        [w1, 0,  1],
        [w1, h1, 1],
        [0,  h1, 1]
    ], dtype=np.float64)

    # Cantos da img2 projetados para o espaço da img1 
    cantos_img2 = np.array([
        [0,  0,  1],
        [w2, 0,  1],
        [w2, h2, 1],
        [0,  h2, 1]
    ], dtype=np.float64)

    # Projeta cada canto da img2 para o espaço da img1 usando H_inv (H mapeia img1→img2)
    H_inv = np.linalg.inv(H)
    cantos_img2_proj = (H_inv @ cantos_img2.T).T
    cantos_img2_proj /= cantos_img2_proj[:, 2:3]  # divide cada linha por seu w

    # ── Bounding box de todos os cantos ──────────────────────────────────────
    todos = np.vstack([cantos_img1[:, :2], cantos_img2_proj[:, :2]])

    x_min = np.floor(todos[:, 0].min()).astype(int)
    y_min = np.floor(todos[:, 1].min()).astype(int)
    x_max = np.ceil(todos[:, 0].max()).astype(int)
    y_max = np.ceil(todos[:, 1].max()).astype(int)

    # ── Offset: garante que todas as coordenadas sejam positivas ─────────────
    # Se x_min < 0, precisamos deslocar tudo em abs(x_min) para a direita
    ox = -x_min if x_min < 0 else 0
    oy = -y_min if y_min < 0 else 0

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    return (canvas_w, canvas_h), (ox, oy)


def warpar_img2(H, img2, canvas_size, offset):
    """
    Distorce a img2 para o espaço da img1, colocando o resultado no canvas.

    Usa mapeamento INVERSO:
      Para cada pixel (x, y) do canvas, calcula de onde ele veio na img2
      aplicando H_inv. Isso evita buracos que surgiriam no mapeamento direto.

    O OpenCV cuida da interpolação bilinear automaticamente via cv2.remap.

    Parâmetros:
      H           : homografia 3×3 (img1 → img2)
      img2        : imagem original (BGR)
      canvas_size : (largura, altura) do canvas
      offset      : (ox, oy) deslocamento aplicado ao canvas

    Retorna:
      img2_warpada : array BGR do mesmo tamanho que o canvas
    """

    canvas_w, canvas_h = canvas_size
    ox, oy = offset

    # ── Monta grade de todos os pixels do canvas 
    # xs e ys são matrizes com as coordenadas de cada pixel no espaço da img1
    xs, ys = np.meshgrid(
        np.arange(canvas_w, dtype=np.float32) - ox,
        np.arange(canvas_h, dtype=np.float32) - oy
    )

    # Coordenadas homogêneas: (x, y, 1) para cada pixel
    ones = np.ones_like(xs)
    coords_canvas = np.stack([xs, ys, ones], axis=0)  # shape (3, H, W)

    # ── Aplica H para mapear canvas (espaço img1) → img2 
    # H mapeia img1→img2, então usamos diretamente (sem inversa)

    # Reshape para (3, H*W), aplica H, volta para (3, H, W)
    coords_flat = coords_canvas.reshape(3, -1)
    coords_img2 = H @ coords_flat
    coords_img2 = coords_img2.reshape(3, canvas_h, canvas_w)

    # Divide por w para voltar a coordenadas 2D
    w_coord = coords_img2[2:3, :, :]
    coords_img2 = coords_img2[:2, :, :] / w_coord

    # ── cv2.remap espera dois mapas float32: um pra x, um pra y
    map_x = coords_img2[0].astype(np.float32)
    map_y = coords_img2[1].astype(np.float32)

    # INTER_LINEAR = interpolação bilinear 
    # BORDER_CONSTANT com 0 → pixels fora da img2 ficam pretos
    img2_warpada = cv2.remap(
        img2, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return img2_warpada


def montar_panorama(img1, img2_warpada, offset):
    """
    Monta o panorama final.

    Lógica simples e direta:
      1. O canvas começa com img2 já distorcida (ela preenche tudo que alcança)
      2. A região de img1 é sobrescrita com img1 original — sem distorção,
         sem máscara, sem mistura. img1 é a base e sempre ganha.

    Isso garante que img1 fique pixel-perfeita no resultado final,
    e img2 preenche apenas o que img1 não cobre.
    """

    ox, oy = offset
    h1, w1 = img1.shape[:2]

    # Canvas começa com img2 warpada
    panorama = img2_warpada.copy()

    # Sobrescreve diretamente com img1 na posição correta
    panorama[oy:oy + h1, ox:ox + w1] = img1

    return panorama


def recortar_panorama(panorama):
    import cv2

    # máscara: True onde é preto (todos canais < 10)
    mascara_preta = np.all(panorama < 10, axis=2).astype(np.uint8) * 255

    # flood fill a partir dos 4 cantos
    # qualquer preto conectado a um canto é "fundo"
    h, w = mascara_preta.shape
    fundo = np.zeros((h + 2, w + 2), np.uint8)  # borda extra que o floodFill exige

    for (cx, cy) in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        if mascara_preta[cy, cx] == 255:           # só faz flood se o canto for preto
            cv2.floodFill(mascara_preta, fundo, (cx, cy), 128)

    # pixels marcados com 128 = fundo confirmado → vira preto total
    # pixels que ficaram 255 = preto DENTRO da imagem (sombra, objeto escuro) → mantém
    fundo_final = (mascara_preta == 128)

    # bounding box do que sobrou
    valido = ~fundo_final
    linhas  = np.any(valido, axis=1)
    colunas = np.any(valido, axis=0)
    y_min, y_max = np.where(linhas)[0][[0, -1]]
    x_min, x_max = np.where(colunas)[0][[0, -1]]

    return panorama[y_min:y_max+1, x_min:x_max+1]


def criar_panorama(H, img1, img2):
    """
    Função principal do módulo — orquestra tudo em sequência.

    Parâmetros:
      H    : homografia 3×3 calculada pelo homography.py
      img1 : imagem base (fica fixa)
      img2 : imagem a ser distorcida

    Retorna:
      panorama    : imagem panorâmica final recortada (BGR)
      canvas_size : (w, h) do canvas usado
      offset      : (ox, oy) offset aplicado
    """

    print("  [warper] Calculando tamanho do canvas...")
    canvas_size, offset = calcular_canvas(H, img1, img2)
    print(f"    Canvas: {canvas_size[0]}×{canvas_size[1]} px  |  Offset: {offset}")

    print("  [warper] Distorcendo img2...")
    img2_warpada = warpar_img2(H, img2, canvas_size, offset)

    print("  [warper] Montando panorama...")
    panorama = montar_panorama(img1, img2_warpada, offset)

    print("  [warper] Recortando bordas...")
    panorama = recortar_panorama(panorama)

    print("  [warper] Concluído.")
    return panorama, canvas_size, offset


# ─────────────────────────────────────────────────────────────────────────────
#  TESTE ISOLADO
#
#  Cria duas imagens sintéticas com uma translação conhecida,
#  calcula H via homography.py e testa o warper.
#
#  > python warper.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from homography import calcular_homografia

    #Imagens sintéticas 
    h, w = 400, 600
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    img2 = np.zeros((h, w, 3), dtype=np.uint8)

    # Fundo colorido diferente para cada imagem
    img1[:, :, 2] = 60   # avermelhado
    img2[:, :, 0] = 60   # azulado

    # Círculos de referência na img1
    refs = [(150, 100), (450, 100), (450, 300), (150, 300)]
    for cx, cy in refs:
        cv2.circle(img1, (cx, cy), 20, (255, 255, 100), -1)

    # Translação de 80px para a direita: esses são os pontos correspondentes na img2
    deslocamento = 80
    pts1 = [[cx, cy] for cx, cy in refs]
    pts2 = [[cx - deslocamento, cy] for cx, cy in refs]

    for cx, cy in pts2:
        cv2.circle(img2, (cx, cy), 20, (100, 255, 255), -1)

    # Calcula H e cria panorama 
    H = calcular_homografia(pts1, pts2)
    print(f"H calculada:\n{np.round(H, 4)}\n")

    panorama, canvas_size, offset = criar_panorama(H, img1, img2)

    #Exibe resultado 
    cv2.imshow("img1 (base)", img1)
    cv2.imshow("img2 (distorcida)", img2)
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()