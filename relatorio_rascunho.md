# Caixa registradora de moedas — esqueleto do relatório

> Esqueleto para vocês reescreverem **com as próprias palavras** (o enunciado zera relatório copiado e pede que o grupo saiba explicar tudo). Trechos entre `<< >>` dependem das fotos reais de vocês. Meta: 5–8 páginas fora capa e referências.

## 1. Introdução
- Problema: contar quanto dinheiro há em moedas fotografadas sobre uma mesa.
- Utilidade: caixa de loja, cofrinho, acessibilidade (pessoa com baixa visão).
- O que o sistema faz: acha cada moeda, descobre o valor, soma e desenha o resultado.

## 2. Fundamentação teórica
**Técnicas das aulas** (explicar o quê / como / por quê): escala de cinza, filtro da mediana e gaussiano, limiarização (Otsu), morfologia (abertura e fechamento), medição de raio.

**Conceito novo 1 — Transformada de Hough para círculos** (≥ 1 página + figura/exemplo à mão):
- Um círculo é `(x−a)² + (y−b)² = r²`. Cada pixel de borda "vota" nos centros possíveis `(a,b)` para cada raio `r`.
- Os centros reais acumulam muitos votos; o acumulador tem picos neles.
- `cv2.HoughCircles` usa a variante por gradiente: a borda aponta para o centro, então só precisa votar ao longo da direção do gradiente (mais barato).
- Parâmetros: `dp`, `minDist`, `param1` (Canny), `param2` (votos mínimos), `minRadius/maxRadius`.
- **Exemplo à mão sugerido:** grade 7×7 com um círculo de raio 2; marcar 8 pontos de borda, desenhar círculos de raio 2 ao redor de cada um e mostrar que se cruzam no centro (acumulador com valor 8).

**Conceito novo 2 — Segmentação por cor em HSV:**
- H = matiz (0–179 no OpenCV), S = saturação, V = brilho. Separar cor de brilho torna a regra mais estável com luz diferente.
- Moeda prata: S baixa. Cobre: H baixo e S alta. Dourada: H ≈ 22, S alta. Bimetálica: centro e anel com saturação muito diferente.
- Funções: `cv2.cvtColor(..., COLOR_BGR2HSV)`; medimos a mediana de H,S,V no centro e no anel.

**Moedas (R$, família atual):** R$0,05 – 22 mm cobre · R$0,10 – 20 mm dourada · R$0,25 – 25 mm dourada · R$0,50 – 23 mm prata · R$1,00 – 27 mm bimetálica. *(Confirmem os diâmetros nas moedas de vocês com um paquímetro/régua e citem a fonte, p.ex. Banco Central.)*

## 3. Metodologia (pipeline)
```
foto → redimensiona (1000 px) → cinza → mediana
     → máscara: |cinza − fundo estimado| + Otsu (×0,4) + morfologia
     → HoughCircles (acha centro e raio) → descarta círculos fora da máscara
     → refina raio (72 raios na máscara, mediana) → remove duplicados
     → HSV centro/anel → material
     → escala (px/mm) a partir das moedas de material único → diâmetro em mm
     → valor = moeda do material com diâmetro mais próximo → soma
```
Tabela de parâmetros (valores em `moedas.py`) e **como chegaram neles** — anotem as tentativas. Experiências que já fizemos e podem entrar como exemplo de ajuste:
- Otsu direto no brilho falhou com iluminação em gradiente → passamos a subtrair o fundo estimado (mediana grande).
- Otsu sozinho perdia moedas prateadas em fundo claro → usamos 0,4 × o limiar de Otsu.
- O raio do Hough variava vários pixels e, em moedas bimetálicas, prendia no círculo interno → refinamento do raio pela máscara.

## 4. Resultados
- `<< ≥15 fotos feitas pelo grupo, incluindo difíceis (luz diferente, moedas encostadas, fundo diferente) >>`
- `<< figura antes/depois de ~3 fotos (pasta resultados/) >>`
- `<< tabela de ≥10 fotos: foto | valor real | valor do programa | acertou? >>` (gerada por `avaliar.py`)
- `<< % de acerto do total, % de acerto da contagem, % de moedas corretas >>`
- `<< matriz de confusão (resultados/matriz_confusao.png) >>`

## 5. Discussão
`<< guardem as fotos que erraram >>`. Hipóteses a verificar com as fotos reais:
- Moedas encostadas (Hough pode fundir/perder uma).
- Reflexo/brilho muda S e H → cobre × dourada confundidas; R$0,10 × R$0,25 dependem da escala.
- Moeda prata em fundo claro (pouco contraste).
- Moedas antigas, desgastadas ou de outra série (R$1,00 com anel/centro invertidos).
- Foto inclinada (círculo vira elipse) → raio errado.
Melhorias: tapete escuro fixo, marcador de escala, k-NN sobre (diâmetro, S, H) treinado nas fotos do grupo, elipses.

## 6. Conclusão
`<< o que aprenderam >>`

## 7. Referências (mín. 3)
- Documentação do OpenCV: tutoriais *Hough Circle Transform* e *Changing Colorspaces*.
- `<< 1 livro da bibliografia da disciplina >>`
- `<< fonte dos diâmetros/materiais das moedas (Banco Central do Brasil) >>`

## 8. Uso de ferramentas de IA
`<< declarar: ex. "Usamos o Claude para gerar a primeira versão do código e o gerador de imagens sintéticas; revisamos e testamos; entendemos cada linha." Seja fiel ao que realmente usaram. >>`
