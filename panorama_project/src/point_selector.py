import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
#  POR QUÊ UMA CLASSE?
#
#  A seleção de pontos tem "memória" — precisa lembrar quais pontos já foram
#  clicados, em qual imagem, quantos pares existem, etc.
#  Uma classe é perfeita pra isso: o `self` guarda todo esse estado.
# ─────────────────────────────────────────────────────────────────────────────

class SeletorDePontos:

    # ─────────────────────────────────────────────────────────────────────────
    #  __init__: chamado quando você faz SeletorDePontos(img1, img2, ...)
    #
    #  Aqui definimos todos os atributos iniciais da classe:
    #  - as imagens
    #  - as listas de pontos
    #  - qual imagem está "esperando" o próximo clique
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, img1, img2, caminho_json="panorama_project/data/points.json"):
        
        self.img1 = img1                  # Imagem da esquerda 
        self.img2 = img2                  # Imagem da direita

        self.pts1 = []                    
        self.pts2 = []                    

        # 1 → próximo clique é na imagem 1
        # 2 → próximo clique é na imagem 2
        self.proximo_clique = 1

        self.caminho_json = caminho_json

        # Paleta de cores para colorir os pares 
        self.cores = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231',
            '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
            '#fabed4', '#469990'
        ]

        # Tenta carregar pontos já salvos antes de abrir a interface
        self._carregar_json()


    # ─────────────────────────────────────────────────────────────────────────
    #  CARREGAR JSON
    #
    #  Se o arquivo já existir (de uma sessão anterior), carrega os pontos.
    #  Isso evita ter que re-clicar tudo toda vez.
    # ─────────────────────────────────────────────────────────────────────────

    def _carregar_json(self):
        if os.path.exists(self.caminho_json):
            with open(self.caminho_json, "r") as f:
                dados = json.load(f)
            self.pts1 = dados.get("pts1", [])
            self.pts2 = dados.get("pts2", [])
            print(f" {len(self.pts1)} pares carregados de '{self.caminho_json}'")
        else:
            print("  Nenhum JSON encontrado. Começando do zero.")


    def _salvar_json(self):
        pasta = os.path.dirname(self.caminho_json)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        dados = {
            "pts1": self.pts1,
            "pts2": self.pts2
        }
        with open(self.caminho_json, "w") as f:
            json.dump(dados, f, indent=2)
        print(f" Pontos salvos em '{self.caminho_json}'")


    # ─────────────────────────────────────────────────────────────────────────
    #  DESENHAR TUDO
    #
    #  Toda vez que um ponto novo é adicionado (ou removido), redesenhamos
    #  as duas imagens do zero. Isso é mais simples do que tentar atualizar
    #  incrementalmente.
    #
    #  O matplotlib converte BGR (OpenCV) para RGB antes de exibir.
    # ─────────────────────────────────────────────────────────────────────────

    def _redesenhar(self):
        # Limpa os dois eixos 
        self.ax1.cla()
        self.ax2.cla()

        # Mostra as imagens 
        self.ax1.imshow(self.img1[:, :, ::-1])
        self.ax2.imshow(self.img2[:, :, ::-1])

        # Se proximo_clique == 2, temos um ponto "pendente" em pts1 sem par ainda
        n_pares = len(self.pts2)         
        n_pts1  = len(self.pts1)        

        #  Desenha os pares completos
        for i in range(n_pares):
            cor = self.cores[i % len(self.cores)]
            x1, y1 = self.pts1[i]
            x2, y2 = self.pts2[i]

            # Círculo na imagem 1
            self.ax1.plot(x1, y1, 'o', color=cor, markersize=8)
            self.ax1.text(x1 + 6, y1 - 6, str(i + 1), color=cor, fontsize=9, fontweight='bold')

            # Círculo na imagem 2
            self.ax2.plot(x2, y2, 'o', color=cor, markersize=8)
            self.ax2.text(x2 + 6, y2 - 6, str(i + 1), color=cor, fontsize=9, fontweight='bold')

        # Se há ponto sem clique na imagem 2
        if n_pts1 > n_pares:
            cor = self.cores[n_pares % len(self.cores)]
            x1, y1 = self.pts1[-1]
        
            self.ax1.plot(x1, y1, '^', color=cor, markersize=10)
            self.ax1.text(x1 + 6, y1 - 6, f"{n_pares + 1}?", color=cor, fontsize=9)

        titulo1 = "Imagem 1"
        titulo2 = "Imagem 2"

        if self.proximo_clique == 1:
            titulo1 += " CLIQUE AQUI"
        else:
            titulo2 += " CLIQUE AQUI"

        self.ax1.set_title(titulo1, fontsize=11, fontweight='bold')
        self.ax2.set_title(titulo2, fontsize=11, fontweight='bold')
        self.ax1.axis('off')
        self.ax2.axis('off')

        instrucoes = (
            f"  Pares completos: {n_pares}  |  "
            "S = salvar    Z = desfazer    Q = sair"
        )
        self.fig.suptitle(instrucoes, fontsize=10, y=0.02)

        self.fig.canvas.draw()


    
    #  HANDLER DE CLIQUE
    def _ao_clicar(self, event):
        # Ignora cliques fora das imagens 
        if event.inaxes not in [self.ax1, self.ax2]:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = round(event.xdata)
        y = round(event.ydata)

        if self.proximo_clique == 1 and event.inaxes == self.ax1:
            self.pts1.append([x, y])
            self.proximo_clique = 2
            print(f" Ponto {len(self.pts1)} na Img1: ({x}, {y})  — agora clique na Img2")

        elif self.proximo_clique == 2 and event.inaxes == self.ax2:
            self.pts2.append([x, y])
            self.proximo_clique = 1
            print(f"  → Ponto {len(self.pts2)} na Img2: ({x}, {y})  — PAR {len(self.pts2)} COMPLETO ")

        else:
            lado = "Imagem 1" if self.proximo_clique == 1 else "Imagem 2"
            print(f" Clique na {lado}!")
            return

        self._redesenhar()


    #  HANDLER DE TECLADO
   
    def _ao_pressionar_tecla(self, event):
        tecla = event.key.lower() if event.key else ""

        if tecla == "s":
            if len(self.pts1) == len(self.pts2) and len(self.pts1) >= 4:
                self._salvar_json()
            else:
                print(f"  Precisa de ≥4 pares completos para salvar. "
                      f"Você tem {len(self.pts2)}.")

        elif tecla == "z":
            # desfazer pontos
            if self.proximo_clique == 2 and len(self.pts1) > len(self.pts2):
                # Remove pontos pendentes na imagem 1
                removido = self.pts1.pop()
                self.proximo_clique = 1
                print(f"Ponto pendente na Img1 removido: {removido}")
            elif len(self.pts2) > 0:
                # Remove o último par completo
                r1 = self.pts1.pop()
                r2 = self.pts2.pop()
                self.proximo_clique = 1
                print(f"   Par {len(self.pts1) + 1} removido: {r1} ↔ {r2}")
            else:
                print("    Nada para desfazer.")
            self._redesenhar()

        elif tecla == "q":
            print("  Encerrando seletor.")
            plt.close(self.fig)


    # ─────────────────────────────────────────────────────────────────────────
    #  INICIAR
    #
    #  Método público que abre a janela e conecta os eventos.
    #  É o único método que o main.py precisa chamar.
    #
    #  Retorna (pts1, pts2) após o usuário fechar a janela.
    # ─────────────────────────────────────────────────────────────────────────

    def iniciar(self):
        plt.rcParams['keymap.save']    = []  
        plt.rcParams['keymap.quit']    = []   
        plt.rcParams['keymap.back']    = []   
        plt.rcParams['keymap.forward'] = []  

        # Cria a figura com dois eixos lado a lado
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.subplots_adjust(bottom=0.08)

        # Conecta os eventos de clique e teclado
        self.fig.canvas.mpl_connect('button_press_event',  self._ao_clicar)
        self.fig.canvas.mpl_connect('key_press_event',     self._ao_pressionar_tecla)

        # Desenha o estado inicial 
        self._redesenhar()

        plt.show()

        # Retorna apenas pares completos
        n = min(len(self.pts1), len(self.pts2))
        return self.pts1[:n], self.pts2[:n]
