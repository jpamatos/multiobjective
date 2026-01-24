# EvoCNN: Evolução de Arquiteturas de Redes Neurais com Algoritmos Genéticos

Este projeto implementa um algoritmo genético para otimizar arquiteturas de redes neurais convolucionais (CNNs). Cada indivíduo da população representa uma rede neural cujo genoma codifica hiperparâmetros estruturais como número de camadas, neurônios, dropout, etc.

## 🚀 Objetivo

Evoluir arquiteturas de redes neurais automaticamente para maximizar métricas como acurácia, F1-score e AUC, utilizando uma abordagem baseada em seleção, crossover e mutação.

## 🧬 Codificação do Genoma

O genoma é um vetor binário com 15 bits. Cada fatia do vetor representa uma parte da arquitetura:

| Gene(s)        | Bits         | Descrição                                   |
|----------------|--------------|---------------------------------------------|
| Conv layers    | 0-1          | Número de blocos convolucionais (1-4)       |
| Neurônios Conv | 2-3          | Número de filtros em potências de 2         |
| Dense layers   | 4-5          | Número de camadas densas (1-4)              |
| Neurônios Dense| 6-11         | Número de neurônios em camadas densas       |
| Dropout        | 13-14        | Taxa de dropout: 0, 0.25, 0.5, 0.75         |

> Obs: o bit 12 está não utilizado no momento (pode ser reservado para uso futuro).

## 🏗️ Estrutura do Projeto

```
├── individual.py       # Implementa a classe Individual
├── main.py             # Roda a evolução genética
├── README.md           # Este arquivo
└── requirements.txt    # Dependências do projeto
```

## 📦 Dependências

- Python 3.9+
- TensorFlow / Keras
- NumPy
- scikit-learn

Instale com:

```bash
pip install -r requirements.txt
```

## 🧪 Executando

```bash
python main.py
```

No `main.py`, você pode definir os dados de treino/teste, população inicial e número de gerações.

## 📈 Métricas Avaliadas

- Acurácia (`accuracy`)
- Loss
- F1-score (macro)
- AUC (multiclasse)
- Latência de predição
- Norma dos pesos

Cada indivíduo guarda essas métricas em um dicionário após o treino.

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para abrir issues ou enviar PRs com melhorias.
