import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from homography import calcular_homografia, erro_reprojecao
from point_selector import SeletorDePontos
from warper import criar_panorama

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))          
DATA_DIR   = os.path.join(BASE_DIR, "data")

IMG1_PATH  = os.path.join(DATA_DIR, "imag1.jpeg")
IMG2_PATH  = os.path.join(DATA_DIR, "imag2.jpeg")
JSON_PATH  = os.path.join(DATA_DIR, "points.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "panorama.jpeg")



#CARREGAR IMAGENS

def carregar_imagens():
    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)

    if img1 is None:
        raise FileNotFoundError(f"Imagem não encontrada: {IMG1_PATH}")
    if img2 is None:
        raise FileNotFoundError(f"Imagem não encontrada: {IMG2_PATH}")

    print(f"  img1: {img1.shape[1]}×{img1.shape[0]} px")
    print(f"  img2: {img2.shape[1]}×{img2.shape[0]} px")

    return img1, img2



#  PASSO 2 — SELECIONAR PONTOS

def selecionar_pontos(img1, img2):
    """
    Abre a interface interativa do SeletorDePontos.
    Se o JSON já existir com pontos salvos, pergunta se quer reutilizá-los.
    """

    if os.path.exists(JSON_PATH):
        resposta = input(f"\n  Pontos salvos encontrados em '{JSON_PATH}'.\n"
                         f"  Usar esses pontos? (s/n): ").strip().lower()
        if resposta == "n":
            os.remove(JSON_PATH)
            print("  JSON removido. Abrindo seletor...")

    seletor = SeletorDePontos(img1, img2, caminho_json=JSON_PATH)
    pts1, pts2 = seletor.iniciar()

    if len(pts1) < 4:
        raise ValueError(f"Mínimo de 4 pares necessários. Você forneceu {len(pts1)}.")

    print(f"\n  {len(pts1)} pares de pontos prontos.")
    return pts1, pts2


# CALCULAR HOMOGRAFIA

def calcular_H(pts1, pts2):
    H = calcular_homografia(pts1, pts2)

    erros, erro_medio = erro_reprojecao(H, pts1, pts2)

    print(f"\n  Homografia calculada:")
    print(np.round(H, 4))
    print(f"\n  Erro de reprojeção por ponto (px): {np.round(erros, 2)}")
    print(f"  Erro médio: {erro_medio:.4f} px")

    if erro_medio > 5.0:
        print("\n Erro alto! Considere rever os pontos selecionados (tecle 'n' na próxima vez).")
    else:
        print("  Homografia com boa qualidade.")

    return H

#CRIAR E SALVAR PANORAMA

def gerar_panorama(H, img1, img2):
    panorama, canvas_size, offset = criar_panorama(H, img1, img2)

    cv2.imwrite(OUTPUT_PATH, panorama)
    print(f"\n  Panorama salvo em: {OUTPUT_PATH}")

    return panorama

#EXIBIR RESULTADO

def exibir(panorama):
    # Redimensiona para caber na tela se for muito grande
    tela_max_w = 1400
    h, w = panorama.shape[:2]

    if w > tela_max_w:
        escala = tela_max_w / w
        panorama_exib = cv2.resize(panorama, (tela_max_w, int(h * escala)))
    else:
        panorama_exib = panorama

    cv2.imshow("Panorama — pressione qualquer tecla para sair", panorama_exib)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img1, img2 = carregar_imagens()
    
    pts1, pts2 = selecionar_pontos(img1, img2)

    H = calcular_H(pts1, pts2)

    panorama = gerar_panorama(H, img1, img2)

    exibir(panorama)

    print("\n  Concluído! Sem erros \n")