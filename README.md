# Processador de Manuscritos Genealógicos — Rol de Confessados

Sistema em Python para extracção automática de dados genealógicos a partir de manuscritos históricos portugueses, usando OCR assistido por inteligência artificial (Google Gemini).

---

## Contexto

Os documentos processados são **Róis de Confessados** (também chamados Róis de Desobriga ou Pascais), registos paroquiais do séc. XVIII que listam os habitantes de uma freguesia organizados por agregados familiares (fogos). Neste caso concreto trata-se da freguesia de **Oliveira de Gumiarães**, com registos do ano **1790**.

Cada página do manuscrito lista os moradores de um conjunto de ruas ou lugares, identificando o chefe de cada fogo, os restantes membros da família, o seu grau de parentesco ou estado civil, e se cada pessoa cumpriu o preceito pascal (confissão anual).

---

## Estrutura do projecto

```
TranscriçãoDeManuscritos/
├── processar_manuscritos.py   # Script principal — processamento completo
├── testar_manuscritos.py      # Script de teste — valida com as primeiras N páginas
├── requirements.txt           # Dependências Python
├── siglas_genealogicas.csv    # Tabela de abreviaturas e siglas históricas (135 entradas)
├── manuscritos/               # Pasta de entrada — PDFs a processar
│   ├── Rol - 1790-2.pdf
│   ├── Rol - 1790-3.pdf
│   └── ...
└── saida/                     # Pasta de saída — resultados gerados automaticamente
    ├── resultados_YYYYMMDD_HHMM.csv
    ├── resultados_YYYYMMDD_HHMM.md
    ├── dados_brutos_YYYYMMDD_HHMM.json
    └── checkpoint.json        # Progresso (apagado após conclusão)
```

---

## Pré-requisitos

- **Python 3.10+**
- **Chave de API do Google AI Studio** — obtida gratuitamente em https://aistudio.google.com/app/apikey

---

## Instalação

```bash
pip install -r requirements.txt
```

Dependências instaladas:
| Pacote | Função |
|---|---|
| `google-generativeai` | Cliente da API Gemini |
| `pymupdf` | Conversão de PDF em imagem (importado como `fitz`) |
| `Pillow` | Manipulação de imagens |
| `python-dotenv` | Leitura do ficheiro `.env` |

---

## Configuração

### 1. Chave de API (obrigatório)

Copia o ficheiro de exemplo e preenche a tua chave:

```bash
cp .env.example .env
```

Edita `.env`:
```
GEMINI_API_KEY=a_tua_chave_aqui
```

A chave nunca é incluída no repositório — o ficheiro `.env` está listado no `.gitignore`.

### 2. Restantes opções

Edita as constantes no topo de `processar_manuscritos.py`:

```python
GEMINI_MODEL        = "gemini-3.1-flash-lite-preview"     # confirmar nome actual em aistudio.google.com
INPUT_FOLDER        = "manuscritos"                       # pasta com os PDFs
OUTPUT_FOLDER       = "saida"                             # pasta para os resultados
PDF_DPI             = 200                                 # resolução de conversão
DELAY_ENTRE_PEDIDOS = 4                                   # segundos entre pedidos à API
```

---

## Utilização

### Processamento completo (todos os PDFs)

```bash
python processar_manuscritos.py
```

### Teste rápido

```bash
python testar_manuscritos.py                                    # testa os primeiros 5 PDFs (padrão)
python testar_manuscritos.py 3                                  # testa os primeiros 3 PDFs
python testar_manuscritos.py nome_do_ficheiro                   # testa um determinado ficheiro
python testar_manuscritos.py nome_do_ficheiro nome_do_ficheiro  # testa de um dado ficheiro a outro
```

O script de teste mostra no terminal, para cada página, os fogos e pessoas identificados com toda a informação extraída, e guarda os resultados em `saida/teste/`.

---

## Formato de saída

O resultado principal é uma tabela com uma linha por pessoa, com as seguintes colunas:

| Coluna | Descrição |
|---|---|
| **Id** | Identificador incremental único por pessoa |
| **Fogo** | Número do agregado familiar (incremental para cada `#` ou `//`) |
| **Lugar** | Rua ou sítio onde reside o fogo |
| **NomeOriginal** | Nome tal como aparece no manuscrito (com abreviaturas) |
| **NomeAtualizado** | Nome com abreviaturas expandidas |
| **Sexo** | M / F |
| **EstadoCivil** | casado/a · solteiro/a · viúvo/a · desconhecido |
| **Parentesco** | cabeça · mulher · filho · filha · criado · criada · oficial · servo · serva · neto · neta · sobrinho · sobrinha · outro |
| **Confessou** | sim · não · ilegível |
| **Observações** | Associação de sub-fogos, separada (sp.), mentecapto, cego, forasteiro, etc. |

São gerados três ficheiros por execução:
- **`.csv`** — separado por `;`, encoding UTF-8 BOM (abre directamente no Excel)
- **`.md`** — tabela em formato Markdown
- **`.json`** — resposta bruta do modelo, útil para depuração e reprocessamento

---

## Como o sistema lê os manuscritos

### Símbolos de fogo

Cada grupo de pessoas constitui um fogo (agregado familiar). Os fogos são identificados por um símbolo antes do primeiro nome:

- **`#`** — cabeça principal do fogo; o homem que chefia o agregado
- **`//`** — sub-fogo: pertence à mesma casa que o `#` imediatamente anterior, mas é um agregado distinto. Pode aparecer manuscrito como `ll`, `‖` ou `11`

O número de fogo é incremental para cada `#` ou `//` encontrado. Os sub-fogos são registados nas Observações com a associação ao fogo principal (ex: *"sub-fogo do fogo 3"*).

### Lugar

O nome da rua ou sítio aparece em texto de maior dimensão, normalmente como cabeçalho entre grupos de pessoas. É propagado automaticamente a todos os fogos das páginas seguintes até que apareça um novo lugar — o sistema mantém memória do último lugar identificado ao longo de todas as páginas.

### Estrutura de cada pessoa

```
# João M.el de Olivr.a cas.     ———— Confes.
  Anna Maria Fon.ca m.er         ———— Confes.
  Jozé f.º                       ————
  Ignacia Tereza f.ª              ———— Confes.
  António Roiz f.al               ———— Confes.
```

Cada linha contém:
1. **Nome** (com abreviaturas históricas)
2. **Papel** — estado civil (`cas.` = casado/a, `S.º` = solteiro/a, `viu.` = viúvo/a) ou parentesco (`m.er` = mulher/esposa, `f.º/f.ª` = filho/filha, `cr.º/cr.ª` = criado/a, `f.al` = oficial)
3. **Confissão** — "Confes." ou "Conf." à direita após linha horizontal indica que a pessoa se confessou; ausência de indicação significa que não confessou

---

## Tabela de siglas (`siglas_genealogicas.csv`)

Contém 135 entradas com abreviaturas, siglas e termos históricos agrupados por categoria:

| Categoria | Exemplos |
|---|---|
| Nomes próprios | `M.el` → Manuel · `Fr.co` → Francisco · `An.to` → António |
| Apelidos | `P.ra` → Pereira · `Oliv.ra` → Oliveira · `Roiz` → Rodrigues |
| Parentesco | `m.er` → mulher · `f.º/f.ª` → filho/filha · `f.al` → oficial |
| Estado civil | `cas.` → casado/a · `S.º` → solteiro/a · `viu.` → viúvo/a |
| Observações | `sp.` → separada (vai para Observações, não EstadoCivil) |
| Tratamentos | `D.` → Dom · `P.e` → Padre · `Dr.` → Doutor |
| Profissões | `cr.º/cr.ª` → criado/criada · `mol.` → moleiro |
| Meses/datas | `7bro` → Setembro · `8bro` → Outubro |

Esta tabela é injectada no prompt enviado ao modelo e pode ser expandida à medida que novas abreviaturas são encontradas nos documentos.

---

## Gestão de limites da API (tier gratuito)

- **Pausa entre pedidos** — espera configurável entre chamadas à API (`DELAY_ENTRE_PEDIDOS`)
- **Retry inteligente** — em caso de erro 429 (rate limit), lê o tempo de espera sugerido pela API e aguarda exactamente esse tempo antes de tentar novamente
- **Checkpoint automático** — após cada página processada, o progresso é guardado em `saida/checkpoint.json`; se a quota diária for atingida, o script pára graciosamente e exporta os resultados parciais com sufixo `_parcial`
- **Retoma automática** — ao correr o script novamente, detecta o checkpoint e pergunta se deve retomar de onde ficou, saltando as páginas já processadas


---

## Notas técnicas

- Os PDFs são ordenados por **ordenação natural** (`Rol-2` antes de `Rol-10`)
- Cada PDF é convertido para imagem em memória (sem ficheiros temporários) e enviado directamente ao modelo via API multimodal
- O modelo devolve JSON estruturado que é parseado e validado; em caso de JSON inválido, são feitas até 3 tentativas antes de ignorar a página
- O estado (Fogo, Lugar) é preservado **entre páginas e entre PDFs distintos** — o modelo recebe o lugar actualmente em vigor como contexto para cada página, devolvendo apenas quando muda
