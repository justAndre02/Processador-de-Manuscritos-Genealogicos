#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processador de Manuscritos Genealógicos — Rol de Confessados
=============================================================
Analisa PDFs de manuscritos históricos (séc. XVII-XVIII) e extrai
dados genealógicos usando o modelo Gemini via API.

Saída: tabela CSV + ficheiro Markdown com colunas:
  Id | Fogo | Lugar | NomeOriginal | NomeAtualizado |
  Sexo | EstadoCivil | Parentesco | Confessou | Observações
"""

import os
import re
import json
import csv
import io
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv  # pip install python-dotenv
import fitz
from PIL import Image
import google.generativeai as genai

# Carrega variáveis do ficheiro .env (se existir)
load_dotenv()

# ====================================================================
# CONFIGURAÇÃO — editar antes de correr
# ====================================================================
API_KEY        = os.getenv("GEMINI_API_KEY", "")  # definido no ficheiro .env
GEMINI_MODEL   = "gemini-3.1-flash-lite-preview"
INPUT_FOLDER   = "manuscritos"              # Pasta com os PDFs de entrada
OUTPUT_FOLDER  = "saida"                    # Pasta para os ficheiros de saída
CSV_MAPPING    = "siglas_genealogicas.csv"
PDF_DPI        = 200                        # Resolução para converter o PDF em imagem
MAX_RETRIES    = 3                          # Tentativas por página em caso de erro
RETRY_DELAY    = 15                         # Segundos de espera base entre tentativas
DELAY_ENTRE_PEDIDOS = 4                     # Segundos de espera entre pedidos bem-sucedidos
CHECKPOINT_FILE = "saida/checkpoint.json"   # Ficheiro de progresso para retomar

# ====================================================================
# ORDENAÇÃO NATURAL DOS FICHEIROS PDF
# ====================================================================
def natural_sort_key(path: Path) -> list:
    """Ordena 'Rol-2', 'Rol-10' correctamente (2 antes de 10)."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", path.name)]

# ====================================================================
# CARREGAMENTO DA TABELA DE SIGLAS
# ====================================================================
def load_siglas(csv_path: str) -> str:
    """Lê o CSV de siglas e formata como bloco de texto para o prompt."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sigla  = row.get("Sigla", "").strip()
            expansao = row.get("Expansão", "").strip()
            uso    = row.get("Significado / Uso", "").strip()
            if sigla and expansao:
                rows.append(f"  {sigla} → {expansao}  [{uso}]")
    return "\n".join(rows)

# ====================================================================
# CONVERSÃO PDF → IMAGEM PIL
# ====================================================================
def pdf_to_images(pdf_path: str, dpi: int = PDF_DPI) -> list[Image.Image]:
    """Converte cada página do PDF numa imagem PIL.Image."""
    images = []
    doc = fitz.open(pdf_path)
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images

# ====================================================================
# CONSTRUÇÃO DO PROMPT
# ====================================================================
def build_prompt(siglas_text: str) -> str:
    return f"""Estás a analisar uma página de um manuscrito histórico português (séc. XVIII) — \
um Rol de Confessados/Desobriga Pascal. É um registo paroquial que lista os habitantes de uma \
freguesia organizados por fogos (agregados familiares).

## ESTRUTURA DO DOCUMENTO

### Símbolos de fogo (aparecem ANTES do primeiro nome do grupo):

**REGRA FUNDAMENTAL**: qualquer símbolo reconhecível antes de um nome (`#`, `//`, `ll`, `11`, `11.`, `‖`) \
cria SEMPRE um novo fogo. Nunca agrupa duas linhas com símbolos no mesmo fogo. \
Linhas SEM símbolo antes do nome pertencem ao fogo mais recentemente aberto.

- **#**  → cabeça principal do fogo (geralmente o homem que chefia o agregado)
- **//** → pertence à mesma casa/propriedade que o fogo # imediatamente anterior, \
mas trata-se de um agregado familiar distinto dentro da mesma habitação. \
O símbolo // pode aparecer como dois traços verticais, "ll", "‖", "11" ou **"11."** (com ponto). \
**VERIFICAÇÃO OBRIGATÓRIA — começo de cada linha**: ANTES de atribuir qualquer parentesco, \
examina SEMPRE o início (extremidade esquerda) de cada linha. Se encontrares dois traços \
verticais consecutivos — seja qual for a forma exacta: `‖`, `//`, `||`, `11`, `11.`, `ll`, `ll.`, `|1`, `1l`, \
dois riscos paralelos — isso é SEMPRE `simbolo="//"` (sub-fogo). \
**Não hesites nem procures alternativas**: a presença de dois traços verticais no início da linha \
é a única prova necessária. Abre imediatamente um novo grupo com `simbolo="//"`. \
**REGRA DE POSIÇÃO — distinção crítica**: \
`//`/`ll`/`11`/`‖` no **início da linha**, ANTES de qualquer nome = marcador de sub-fogo; \
`Ir.`/`Ir.ª` APÓS o nome = parentesco irmão/irmã. \
Se vires `11`, `11.`, `ll` ou `‖` **antes** de um nome (início de linha), é SEMPRE `//` (sub-fogo), \
NUNCA `Ir.` (irmão). O `1` cursivo parece `I` e o segundo `1` parece `r`, mas a posição antes \
do nome é o indicador inequívoco de que se trata do marcador de sub-fogo. \
**O primeiro nome após `//` é SEMPRE a cabeça do sub-fogo** (`parentesco = "cabeça"`), \
mesmo que essa pessoa só tenha estado civil indicado (`S.ª`, `V.ª`, `cas.`) e nenhum \
outro marcador de parentesco. \
**`//` tem prioridade sobre `f.al`/oficial**: se a pessoa tiver `//` + `f.al`, \
usa `parentesco = "cabeça"` e coloca `"oficial"` em `observacoes`. \
**`//` (ou `#`) seguido de `P.e` / `o P.e`**: o símbolo de fogo é `//` (ou `#`); \
o `P.e` / `o P.e` NÃO é parte do marcador de fogo nem do nome — vai OBRIGATORIAMENTE \
para `observacoes` como `"Padre"`. Exemplo: `11 o P.e Thomaz Fran.co` → \
`simbolo="//"`, `nome_original="Thomaz Fran.co"`, `observacoes="Padre"`. \
**NUNCA atribuas `"irmão"`/`"irmã"` a uma pessoa que tenha marcador `//`**, nem por omissão. \
**REGRA ABSOLUTA — `11.`/`11`/`//`/`ll` ANTES do nome = SEMPRE `simbolo="//"`, SEM EXCEPÇÃO**: \
se o início da linha (extremidade esquerda) mostrar `11.`, `11`, `//`, `ll` ou qualquer variante \
de dois traços verticais, devolve **obrigatoriamente** `simbolo="//"`. \
NUNCA devolvas `simbolo="#"` para um grupo cujo início de linha tenha `11.`/`//`. \
**MÚLTIPLOS SUB-FOGOS CONSECUTIVOS**: quando vários grupos com `//`/`11.` aparecem um a seguir \
ao outro depois de um `#`, são TODOS sub-fogos do MESMO `#` — NÃO são sub-fogos uns dos outros. \
Exemplo concreto (3 sub-fogos do mesmo `#`): \
`# M.el Caetano (Padre)`, `11. Ianuario Iozé V.o`, `11. D. Catharina V.ª`, `11. R.do Iozé P.e` → \
todos têm `simbolo="//"`, todos são cabeças dos seus sub-fogos, todos são sub-fogos do `#` original. \
Mesmo que pareça que o terceiro sub-fogo começa uma nova secção: se a linha começa por `11.`/`//`, \
é SEMPRE `simbolo="//"`, nunca `"#"`.
- **~**  → continuação do fogo anterior (o fogo começou na página passada e continua nesta). \
Usa `~` **exclusivamente** para o **primeiro grupo da página** quando a lista de pessoas \
recomeça sem qualquer símbolo `#` visível — ou seja, a página começa a meio de um fogo. \
**CRÍTICO — marcadores de parentesco num grupo `~` aplicam-se EXACTAMENTE como em qualquer outro fogo**: \
`cr.`/`cr.ª`/`cz.`/`cz.ª` = criado/criada; `f.`/`f.ª` = filho/filha; `m.er`/`mul.` = mulher; \
`Ir.`/`Ir.ª` = irmão/irmã. **NUNCA uses `"desconhecido"` quando o marcador está escrito** — lê \
cada entrada individualmente nesta página e aplica a regra normalmente, independentemente de o fogo \
ter começado na página anterior. A primeira pessoa de um grupo `~` **NÃO é obrigatoriamente \
cabeça** — usa o parentesco que estiver escrito no manuscrito.

### Lugar / Rua:
- Aparece em texto maior, centrado ou em cabeçalho, ENTRE grupos de pessoas.
- Identifica a rua, sítio ou lugar onde residem os fogos que se seguem.
- Nas primeiras páginas pode haver um cabeçalho geral com o nome da freguesia e o ano \
(ex: "Arcela e Canno. 1790").
- **"Doutra parte"**: se aparecer em letras maiores a meio da página, \
significa "do outro lado da rua/caminho". A forma completa é sempre **"Doutra Parte"** \
(nunca "Doutra Part" ou variantes incompletas). Nesse caso, \
`lugar_novo` do primeiro fogo a seguir passa a ser o **lugar actualmente em vigor** \
(o que consta no CONTEXTO DESTA PÁGINA ou o último `lugar_novo` definido antes deste ponto) \
com o sufixo **" - Doutra Parte"**. \
**IMPORTANTE**: usa sempre o lugar em vigor *neste momento da página*, \
nunca um lugar de páginas anteriores. \
Exemplos: se o lugar em vigor é "Canno" → "Canno - Doutra Parte"; \
se é "Santa Cruz" → "Santa Cruz - Doutra Parte"; \
se é "Oliveiras" → "Oliveiras - Doutra Parte".
- **ATENÇÃO paleográfica**: o **C maiúsculo** e o **O maiúsculo** podem confundir-se \
com **L**. Os nomes de lugares desta freguesia incluem: \
Arcela, Canno (não "Lanno"), Carreira, Corredoura, Cruz, Fonte, Lameiro, Leira, \
Oliveiras (não "Liveiras"), Outeiro, Quintã, Ribeira, Rio, Rua, Vale, Vila. \
Se leres "Lanno" reconsidera "Canno"; se leres "Liveiras" reconsidera "Oliveiras". \
**Esta confusão C/L aplica-se também a apelidos**: se extraíres um apelido que não pareça \
português (ex: "Copey"), reconsidera — muito provavelmente o `C` inicial é um `L` cursivo \
e o apelido real é um apelido português comum (ex: "Lopez" → expande para "Lopes"). \
**`L` cursivo maiúsculo — traço descendente abaixo da linha (meia-lua)**: o `L` maiúsculo \
nestes manuscritos tem um traço característico que desce abaixo da linha de base, formando \
uma curva em meia-lua para baixo. Este traço é frequentemente confundido com `f`, `j`, `g` \
ou outras letras com prolongamento inferior. \
**Regra de verificação obrigatória**: se a leitura de uma palavra produzir um resultado que \
**não existe em português nem em português arcaico** (ex: "fexua", "jexua", "geyua"), \
reconsidera imediatamente — a primeira letra é quase certamente um `L` cursivo mal lido. \
Exemplo concreto: `"fexua"` não existe em português → é `"Leyva"` (apelido, expande para "Leiva"). \
Esta verificação aplica-se a qualquer palavra do manuscrito: nome, apelido ou topónimo.

### Estrutura de cada pessoa dentro do fogo:
1. **Nome** (frequentemente com abreviaturas — ver tabela abaixo)
2. **Papel no fogo**: estado civil (casado/a, solteiro/a, viúvo/a) OU \
parentesco com a cabeça (m.er/mul. = mulher; f.º/f.ª = filho/filha; \
cr.º/cr.ª = criado/criada; etc.)
3. **À direita, após uma linha horizontal** (campo `confessou`): \
- "Confes." / "Conf." → `confessou = "sim"`. \
- Qualquer outra palavra ou abreviatura nessa posição (ex: "D.M.", "Saptis", "Soptis", "Sept.") → `confessou = "sim"` \
  **e** acrescenta essa palavra/abreviatura em `observacoes`. \
- Ausência de qualquer texto após a linha horizontal → `confessou = "não"`. \
**`F.`** que apareça **acima ou ao longo da linha** (como anotação separada) significa que a \
pessoa **faleceu** — coloca "falecida" (sexo F) ou "falecido" (sexo M) em `observacoes`. \
**`pg` / `pag`** que apareça **acima da linha horizontal** ou como anotação separada: significa que a pessoa pagou — coloca `"pagou"` em `observacoes`. \
`F.` é independente de "Confes.": os dois podem coexistir na mesma linha; \
`confessou` segue a regra definida acima (qualquer texto = "sim"; ausência = "não"). \
**IMPORTANTE**: `F.` = falecimento NÃO implica viúvo/a — o `estado_civil` deve reflectir \
exactamente o que está escrito antes da linha horizontal (S.ª = solteira, cas. = casada, etc.).
4. **Abreviaturas de papel/parentesco** que aparecem depois do nome (não fazem parte do nome): \
m.er/mul. = mulher (esposa); f.º/f.°/f. = filho; f.ª/f.a = filha; \
**CRÍTICO — `m.er`/`mul.` NUNCA é "mãe"**: `m.er` e `mul.` têm as letras `e`+`r` explícitas após o `m` \
e produzem SEMPRE `parentesco="mulher"` (esposa) — **NUNCA "mãe"**. \
As formas de "mãe" (`Mãe`, `mãi`, `May`, `Mai`, `Mae`, `Maj`) são palavras escritas por extenso — \
NUNCA abreviaturas com `e`+`r`. Se vires `m.er` ou `mul.` após um nome feminino, \
é OBRIGATORIAMENTE `parentesco="mulher"` (esposa), independentemente do contexto. \
cr.º/cr.ª/cr. = criado/criada (qualquer forma abreviada de cr seguida de ponto ou superscript); \
**CRÍTICO — forma feminina `cr.ª` / `cr.a` / `cz.ª`**: a criada tem sempre marcador `cr.ª` ou `cr.a` \
(superscript `ª` ou `a` minúsculo elevado após `cr`); o `r` cursivo parece `z`, tornando `cr.ª` em \
`cz.ª` ou `cza.` — TODAS estas formas = `parentesco="criada"`. \
O superscript `ª` é pequeno mas NUNCA deve ser ignorado; `cr`/`cz` seguido de terminação elevada \
feminina é SEMPRE criada. **`cr.ª`/`cz.ª` nunca é estado civil** — é SEMPRE parentesco "criada". \
**REGRA CRÍTICA — `parentesco="mulher"` EXIGE `m.er`/`mul.` explícito no manuscrito**: \
NUNCA atribuas `parentesco="mulher"` a uma pessoa só porque é do sexo F e tem `estado_civil="casada"`. \
Uma criada casada, uma irmã casada ou outra parente casada NÃO é a esposa — o marcador `m.er`/`mul.` \
tem de estar escrito. Se não está escrito `m.er`/`mul.`, usa o parentesco que estiver indicado \
(cr. = criada, Ir.ª = irmã, etc.) ou `"desconhecido"` se não houver indicação. \
**ATENÇÃO paleográfica — `cr.` vs `caz.`/`cas.`**: `cr.` começa por `c`+`r`; `caz.`/`cas.` começa por `c`+`a`. \
Se vires `c` + letra seguinte após um nome, verifica se a segunda letra é `r` (criado) ou `a` (casado). \
**CRÍTICO — `cz.` após um nome = criado/criada, NUNCA casado/casada**: nestes manuscritos o `r` cursivo \
assemelha-se ao `z`, por isso `cr.` é frequentemente lido como `cz.`. A distinção é o NÚMERO DE LETRAS \
antes do ponto: `cr.`/`cz.` tem DUAS letras (c+r/z); `caz.`/`cas.` tem TRÊS letras (c+a+z/s). \
Se a abreviatura após um nome tiver apenas DUAS letras a começar por `c`, é SEMPRE `cr.` (criado/criada) — \
NUNCA `caz.` (casado/casada). Exemplos: `"Manoela cz."` → `parentesco="criada"`, não casada. \
`"Luis Jozé cr."` → `parentesco="criado"`, `estado_civil="solteiro"` (por omissão). \
`cr.` NUNCA implica "casado"; criados sem estado civil explícito são SEMPRE solteiros/solteiras. \
**`cr.` vs `f.` (filho)**: `cr.` tem DUAS letras distintas (`c`+`r`); `f.` filho é UMA única letra `f`. \
Se a abreviatura após um nome tem duas letras começando por `c`, é `cr.` (criado) — NUNCA `f.` (filho). \
`"Luis Jozé cr."` → `parentesco="criado"`, NÃO filho. \
**Ir./ir./Ir.º/ir.º = irmão; Ir.ª/ir.ª = irmã** (aparecem depois do primeiro nome, não são apelido); \
**`Ir.` vs `f.` (filho)**: `Ir.` tem DUAS letras (`I`+`r`); `f.` filho é UMA única letra `f`. \
Se a abreviatura após um nome tem duas letras começando por `I`/`i`, é `Ir.` (irmão) — NUNCA `f.` (filho). \
`"Joaquim Fr.co de Leyva Ir."` → `parentesco="irmão"`, NÃO filho.
f.al = oficial **apenas para homens** (grau de parentesco — usa `parentesco="oficial"` e não coloques em `observacoes` \
a não ser que a pessoa seja também a cabeça do fogo, que nesse caso o `parentesco="cabeça"` e `"oficial"` vai para `observacoes`); \
**REGRA CRÍTICA**: se `f.al` aparecer após um nome **feminino** (sexo=F), NÃO é "oficial" — \
é quase sempre uma leitura errada de `f.ª` (filha); usa `parentesco="filha"` e **não** colocas nada em `observacoes`; \
se a linha tiver `//`, vai para `observacoes` e `parentesco = "cabeça"`); \
sobr.º/sobr.ª/sobr.o/sobr.a = sobrinho/sobrinha; afilh.º/afilh.ª/afilh.o/afilh.a = afilhado/afilhada; serv./servo/serva = servo/serva; etc. \
**ATENÇÃO paleográfica — `Ir.` vs `Sz.`**: nestes manuscritos o `I` cursivo assemelha-se a um `S` \
e o `r` cursivo pode parecer `z`, fazendo com que `Ir.` (irmão) seja lido como `Sz.` (Sousa). \
**Regra contextual obrigatória**: os membros de um fogo partilham o apelido da cabeça. \
Se o cabeça do fogo NÃO tiver "Sousa" como apelido, então qualquer `Sz.` que apareça \
depois do primeiro nome de um membro NÃO é o apelido Sousa — é `Ir.` (irmão) mal lido. \
`Sz.`/`Sz.ª` como apelido só é válido se o cabeça do mesmo fogo TAMBÉM tiver "Sousa". \
`Ir.`/`Ir.ª` é SEMPRE parentesco (irmão/irmã) e NUNCA faz parte do nome. \
Exemplo: fogo cujo cabeça é "João da Costa" → `"Bento Sz."` = `nome_original="Bento"`, `parentesco="irmão"`. \
**Oficial**: Palavras ou abreviaturas como `ofecial`, `oficial`, `ofecial.` ou `oficial.` e `f.al` NUNCA fazem parte do nome. Usa `parentesco="oficial"` (a menos que a pessoa seja cabeça de casal, caso em que vai para observações). \
**IMPORTANTE**: qualquer variante de "f" abreviado (`f.º`, `f.°`, `f.`, `fo.`) \
isolado após o nome é SEMPRE parentesco **"filho"** — mesmo que outras abreviaturas \
(estado civil, observações) apareçam a seguir. \
**Excepções ao IMPORTANTE** (não são `f.` filho): \
`Frz.` / `Frz` / `Friz` = apelido **Fernandes**; \
`Ir.` / `ir.` / `Ir.ª` / `ir.ª` = **irmão/irmã** — o `I` cursivo pode parecer `f`, mas `Ir.` tem SEMPRE duas letras (`I`+`r`), enquanto `f.` filho é uma única letra `f` isolada. \
Nunca uses `"filho"` quando a abreviatura tiver duas ou mais letras (ex: `Ir.`, `Frz.`). \
**Atenção inversa de erro de leitura**: se a imagem contiver APENAS `"f.o"` (ou `f.`) o modelo não pode confundir com `Ir.` (irmão) ou ignorá-lo; `f.o` significa sempre parentesco **"filho"**. O `I` e `r` não constam lá. \
**Sequência colada `f.oDem.te` (ou `f.o Dem.te`)**: interpreta obrigatoriamente como duas siglas: \
`f.o` = parentesco `"filho"` e `Dem.te` = observação `"demente"`. \
NUNCA converter `Dem.te` em `"menor"` neste contexto. \
Exemplo concreto: a linha **"João da Costa f.° caz. Sep."** deve produzir \
`nome_original="João da Costa"`, `parentesco="filho"`, `estado_civil="casado"`, \
`observacoes="separado"` — o `f.°` define o parentesco, `caz.` o estado civil, `Sep.` a observação.
5. **Abreviaturas de estado civil** que aparecem depois do nome (não fazem parte do nome): \
cas./caz./cass. = casado/a; S.º/soltr./solt. = solteiro/a; \
viu./viu.o/viuva/V.o/V.º/V.°/V.a/V.ª = viúvo/viúva. \
**ATENÇÃO — `Vr.a` / `Vr.ª` / `V.ra` são apelido `Vieira` (nome), NÃO estado civil**: \
se o token tiver `r` depois do `V`, trata como apelido e expande para `"Vieira"` no nome, \
nunca para `estado_civil="viúvo/viúva"`. \
**ATENÇÃO — `S.ª`/`S.a` são ambíguas** (distingue pela preposição que as precede): \
— Com `da`/`de`/`dos`/`das` imediatamente antes (ex: `da S.ª`, `da S.a`) = apelido **Silva** → faz parte do nome; nunca é estado civil neste caso. \
— **SEM preposição antes** (ex: `"Luiza Maria S.a"`, `"Tereza Barbara S.a"`) = **solteira** → estado civil; \
  NUNCA é o apelido Silva — mesmo que outros membros do fogo se chamem Silva. \
Exemplos: `"Manoel da S.ª"` → `nome_expandido="Manuel da Silva"`, `estado_civil` pelo contexto. \
`"Luiza Maria S.a"` → `nome_expandido="Luísa Maria"`, `estado_civil="solteira"`. \
`"Maria Tereza S.a Ir.ª"` → `nome_expandido="Maria Teresa"`, `estado_civil="solteira"`, `parentesco="irmã"`. \
`"Tereza Barbara S.a"` → `nome_expandido="Teresa Bárbara"`, `estado_civil="solteira"` (o `S.a` sem `da` = solteira, não Silva). \
**Atenção paleográfica**: nestes manuscritos o `V` cursivo pode assemelhar-se a um `S`. \
Se a letra antes do ponto/superscript for ambígua entre `V` e `S`, e o contexto \
(fogo sem mulher, pessoa mais velha, cabeça de sub-fogo) sugerir viuvez, prefere `"viúvo/a"`. \
**REGRA CRÍTICA — `V.ª` / `V.a` / `V.o` / `V.º` / `V.°`**: quando esta abreviatura aparece \
explicitamente escrita após o nome, é SEMPRE "viúva" ou "viúvo" — NUNCA "casada/casado". \
Não uses "casado/a" para uma cabeça que aparece explicitamente marcada com `V.ª`/`V.o`. \
**`V.º`/`V.ª` aplica-se a QUALQUER pessoa — não só à cabeça**: um criado, criada, irmão, \
irmã, filho, filha com `V.º`/`V.ª` explícito tem SEMPRE `estado_civil="viúvo"`/`"viúva"`, \
independentemente do parentesco. A regra de omissão (criado → solteiro) só se aplica \
6. **Observações especiais** que aparecem depois do nome e vão para o campo Observações \
(não para os outros campos): sp. / Sep. / sep. / Se.p / se.p = separado/separada; \
mentecapto; cego; mudo; forasteiro; deslocado; menor (m.); ausente (aus.te, auz.te); demente; [ilegível]; etc. \
**ATENÇÃO paleográfica — `Sego` mal lido como `Lego`**: nestes manuscritos o `S` maiúsculo cursivo parece um `L`. Se leres a palavra `"Lego"` junto ao nome de alguém (ex: antes de `cr.`), trata-se na verdade de `"Sego"` (grafia antiga para Cego). Não faz parte do nome! Retira do nome e coloca obrigatoriamente `"cego"` em `observacoes`. \
**`Mudo` / `mudo`** que apareça junto ao nome: indica que a pessoa é muda. Não faz parte do nome! Retira obrigatoriamente do nome e coloca `"mudo"` ou `"muda"` em `observacoes`. \
**`Ama`** que apareça junto ao nome de uma criada indica a sua função de ama (ama de leite \
ou ama seca) — o `parentesco` mantém-se **"criada"** e regista `"Ama"` em `observacoes`. \
**`Aja` / `Aia`** que apareça junto ao nome (frequentemente com ponto no fim "Aja.") indica \
a função de aia — o `parentesco` DEVE ser **"criada"** e regista obrigatoriamente `"Aia"` \
em `observacoes`. Aja/Aia NUNCA faz parte do nome. \
**`Escrava` / `Escrav.` / `Escravo`** junto ao nome: a pessoa é escrava → coloca "escrava" (F) \
ou "escravo" (M) em `observacoes`; `parentesco` mantém-se "criada"/"criado". \
**ATENÇÃO paleográfica — `Subd.` em fim de linha**: Se a linha apresentar no fim a palavra `Subd.` ou `fubd.`, nunca a transformes em "in minoribus", em "Subdiácono" ou algo parecido. Coloca literalmente a sigla com um ponto de interrogação nas observações: `"Subd.?"` ou `"fubd.?"`. \
**ATENÇÃO paleográfica — `Escrava` lida como `Cicrava`**: o `E` cursivo maiúsculo parece `C`, \
e o `s` parece `i`, tornando "Escrava" em "Cicrava". "Cicrava" não existe em português — \
se leste "Cicrava" ou forma semelhante, é SEMPRE "Escrava".
**`Dem.te` = demente (não menor)**: quando aparecer `Dem.te` (isolado ou colado a `f.o`), \
regista `"demente"` em `observacoes`. Não substituir por `"menor"`.
7. **Sequências de abreviaturas junto ao nome**: quando várias abreviaturas aparecem \
consecutivamente após o nome (ex: `sobr.º V.º F.`), analisa cada uma independentemente \
e preenche o campo correcto para cada: \
`sobr.º`/`sobr.ª`/`sobr.o`/`sobr.a` = parentesco **sobrinho/sobrinha**; \
`V.º`/`V.ª`/`V.o`/`V.a` = estado civil **viúvo/viúva**; \
`F.` (acima ou junto) = **falecido/falecida** em `observacoes`. \
Exemplo: `"Bernardo Boa Ventura sobr.º V.º F."` → \
`parentesco="sobrinho"`, `estado_civil="viúvo"`, `observacoes="falecido"`. \
Nunca ignores abreviaturas só porque estão agrupadas — cada uma tem o seu campo destino. \
**`F.` pode aparecer no FINAL de qualquer sequência de marcadores** (incluindo após `V.º`, `cunh.`, \
ou outros): ex: `11 Ianuario Iozé Montr. V.o F.` → `estado_civil="viúvo"`, `observacoes="falecido"` (M). \
O `F.` NUNCA se perde por aparecer a seguir a outro marcador — procura-o SEMPRE até ao fim da linha.

### Tabela de abreviaturas — usa esta para expandir nomes e termos:
{siglas_text}

---
## TAREFA

Extrai TODAS as pessoas visíveis na imagem e devolve EXCLUSIVAMENTE um JSON \
válido com a estrutura seguinte (sem texto antes ou depois, sem blocos de código markdown):

{{
  "lugar_atual": "novo lugar se aparecer ANTES do primeiro fogo da página, caso contrário null",
  "lugar_proximo": "nome do novo lugar se um indicador de lugar aparecer DEPOIS do ÚLTIMO fogo da página (fim de página, a anunciar a secção seguinte), caso contrário null",
  "grupos": [
    {{
      "simbolo": "# | // | ~",
      "lugar_novo": "nome do novo lugar se um indicador de lugar aparecer IMEDIATAMENTE ANTES DESTE fogo, caso contrário null",
      "pessoas": [
        {{
          "nome_original": "APENAS o nome próprio e apelido, exactamente como aparece no manuscrito — SEM abreviaturas de papel/estado civil",
          "nome_expandido": "APENAS o nome próprio e apelido com abreviaturas do nome expandidas — SEM palavras de papel ou estado civil",
          "sexo": "M ou F",
          "estado_civil": "casado | casada | solteiro | solteira | viúvo | viúva | desconhecido",
          "parentesco": "cabeça | mulher | filho | filha | irmão | irmã | primo | prima | criado | criada | oficial | aprendiz | companheiro | mestre | caixeiro | servo | serva | neto | neta | sobrinho | sobrinha | afilhado | afilhada | sogro | sogra | genro | nora | cunhado | cunhada | tio | tia | avô | avó | pai | mãe | outro | desconhecido",
          "confessou": "sim | não | ilegível",
          "observacoes": "notas relevantes separadas por vírgula, vazio se não houver"
        }}
      ]
    }}
  ]
}}

## REGRAS ADICIONAIS
1. **Nome**: `nome_original` e `nome_expandido` contêm APENAS o nome próprio e apelido. \
Nunca incluas abreviaturas de papel (m.er, f.º, f.°, f., cr.º, f.al, sobr.º, **P.e**, etc.) nem de \
estado civil (S.º, cas., viu., etc.) no nome — essas vão para as colunas correctas. \
**`P.e` / `o P.e` NUNCA faz parte do nome** — vai OBRIGATORIAMENTE para `observacoes` como `"Padre"` (ver Regra 7). \
**Atenção paleográfica — `oP.e` mal lido como `ob.`, `ol.` ou `R.do`**: o prefixo `oP.e` \
pode assemelhar-se a `ob.`, `ol.` ou até `R.do` devido à grande laçada do 'P'. Se a primeira palavra antes de um nome for lida como `ob.`, `ol.` ou `R.do`, é quase certamente `oP.e` (o Padre). Tem de ser removido do nome e colocado como `"Padre"` nas `observacoes`. \
**Atenção paleográfica — `oP.e Sacristão Ant.o` mal lido como `D. Iacinta Ant.a`**: \
em algumas linhas o conjunto `oP.e` + `Sacristão` é indevidamente lido como `D.` + `Iacinta`. \
Nestes casos, `Iacinta` não é nome: deve ir para `observacoes` como `"Sacristão"` e o título \
eclesiástico deve ser `"Padre"`. Além disso, quando o nome real é `Ant.o` (António), \
nunca converter para `Ant.a` (Antónia).
**ATENÇÃO ESPECIAL**: `Pr.ª` / `Pr.a` / `P.ª` / `P.ra` são abreviaturas do apelido **Pereira** \
e fazem SEMPRE parte do nome — nunca as omitas, nunca as interpretes como parentesco. \
Se o manuscrito tiver "Anna Pr.ª", `nome_original` = "Anna Pr.ª" e `nome_expandido` = "Anna Pereira". \
**Atenção paleográfica — `Pr.ª` vs `S.a` / `da S.a`**: nestes manuscritos `Pr.ª` (Pereira) é \
frequentemente confundido com `S.a` ou até `da S.a` (Silva) porque: \
o `P` cursivo com traço descendente parece `d`; o `r` parece `a`; e `.ª` parece `S.a`. \
Assim `Pr.ª` inteiro pode ser lido como `da S.a`. \
**Se a leitura produz "da Silva" mas NÃO há um `da` claramente visível antes da \
abreviatura no manuscrito (o `d` que leste é na realidade o `P` cursivo de `Pr.ª`), \
reconsidera obrigatoriamente: é muito provável que seja `Pr.ª` (Pereira). \
Se `da` está claramente escrito antes de `S.ª`/`S.a` no manuscrito, é "da Silva" — \
mesmo que outros membros do mesmo fogo não tenham o apelido Silva. \
**Além disso, `Pr.ª`/`Pr.a` pode ser mal lida como `de Beca`, `dBec.a` ou `de Eça`** \
porque o `P` cursivo parece `d`/`B` e `.a`/`.ª` parece `ca`/`ça`. \
Se leste `"de Beca"` ou formas similares para um apelido sem `de` claramente visível, \
reconsidera: é quase certamente `Pr.ª` = **Pereira**. \
Exemplo: `"Iozé Pr.a"` → `nome_expandido="José Pereira"` (não "José de Beca").** \
Da mesma forma, `Frz.` / `Frz` / `Friz` são abreviaturas do apelido **Fernandes** e fazem SEMPRE \
parte do nome — nunca confundas com `f.º`/`f.°`/`f.` (parentesco filho). \
`Frz.` começa por maiúscula e tem mais letras; `f.` parentesco é apenas "f" minúsculo isolado. \
**ATENÇÃO — `Fran.º`/`Fran.co`/`Fr.co` NÃO são `Frz.` (Fernandes)**: \
`Fran.º` e `Fran.co` são abreviaturas do nome próprio **Francisco** — o `an` cursivo pode parecer `z`, \
mas se vires `Fr` seguido de `an`/`co`/`º` é sempre Francisco; `Frz.` tem apenas `Fr`+`z` sem mais letras. \
**ATENÇÃO — `Ferr.a` / `Fer.ª` / `Frr.ª` / `Ferr.ª` NÃO são `Frz.` (Fernandes)**: \
estas formas com `rr` duplo são apelido **Ferreira** — o `rr` cursivo pode parecer `rz`, \
mas `Ferr.a`/`Frr.ª` têm SEMPRE dois `r`. \
Se leste `Ferz.` ou `Frr.` após um nome próprio, reconsidera: é quase certamente `Ferr.a`/`Ferr.ª` = **Ferreira**; \
`Frz.` (Fernandes) tem apenas um `r` e o segundo caractere é `z`. \
**DISTINÇÃO CRÍTICA `Frr.a`/`Fr.a` (Ferreira) vs `Fr.co`/`Fr.º` (Francisco)**: \
`Fr.co`/`Fran.co`/`Fr.º` são nome próprio **Francisco** — têm sempre `co` ou `º` depois de `Fr`; \
`Frr.a`/`Frr.ª`/`Fr.a`/`Ferr.a` são apelido **Ferreira** — terminam sempre em `a` (não em `co`/`º`). \
Se após um nome próprio vires uma abreviatura que começa por `Fr` e **termina em `a`**, é SEMPRE \
**Ferreira** (apelido), nunca Francisco. Exemplo: `"Antonio Frr.a"` → `nome_original="Antonio Frr.a"`, \
`nome_expandido="António Ferreira"` — NUNCA `"António Francisco"`. \
**`M.do` / `Mach.do` = apelido **Machado`**: se aparecer após o nome próprio, faz SEMPRE \
parte do nome e DEVE ser incluído em `nome_original` e expandido para "Machado" em `nome_expandido`. \
Nunca o omitas nem o interpretes como parentesco ou estado civil. \
**Atenção paleográfica — `M` vs `N` cursivo**: o `M` maiúsculo cursivo pode assemelhar-se \
ao `N` — se leste `N.do` após um nome, é quase certamente `M.do` = **Machado**. \
**`Px.to` / `Px.ª` / `Pix.to` = apelido **Peixoto`**: se aparecer após o nome próprio, faz SEMPRE \
parte do nome e DEVE ser incluído em `nome_original` e expandido para "Peixoto" em `nome_expandido`. \
Nunca o interpretes como parentesco, estado civil ou outro campo. \
**Grafia `Preira` / `Pr.ira` = apelido **Pereira`**: se leres "Preira" (faltando o primeiro "e") \
ou formas semelhantes como `Pr.ira`, trata-se de Pereira — **nunca o expandas para Peixoto**. \
Exemplo: `"Thomé Pereira"` → `nome_expandido="Tomé Pereira"`. \
**Nomes com `D.`/`D.ª` (Dom/Dona) — lê o nome completo**: pessoas com prefixo honorífico \
`D.` ou `D.ª` têm frequentemente dois ou três nomes próprios e/ou apelido (ex: `"D. Anna M.a Leonor"`). \
O `D.` ou `D.ª` DEVE constar no `nome_original` e DEVE ser EXPLICITAMENTE expandido no `nome_expandido` \
para "Dom" ou "Dona" (ex. `"Dona Jacinta Margarida"`). \
**Nomes com "pelo Crisma" / "pela Crisma"**: se a expressão "pelo Crisma" ou "pela Crisma" (que muitas vezes é lida erroneamente em cursivo como "pela Cierma", "pela Creyma", etc.) aparecer a separar dois nomes (ex: `"Custodia pela Crisma Maria"` ou `"Custodia, pela Crisma Maria"`), significa que a pessoa mudou de nome na Confirmação. \
Nesse caso, `nome_original` e `nome_expandido` devem manter APENAS o primeiro nome (ex: `"Custodia"` / `"Custódia"`), e deves adicionar a anotação do novo nome nas `observacoes` (ex: `"pelo Crisma Maria"`). Não incluas a expressão "pela Crisma Maria" na coluna de nome. \
**Excepção obrigatória**: `D.or` / `D.tor` / `Dr.` / `o D.tor` / `oD.or` / `Dor` = **Doutor** (título) e NUNCA faz parte do nome; \
nestes casos remove o título do `nome_original` e `nome_expandido` e coloca obrigatoriamente `"o Doutor"` ou `"Doutor"` em `observacoes`. \
**NUNCA omitas este marcador inicial nem o percas**: se existir um token antes do nome com forma próxima de `Dor`/`D.or`/`Dr`/`D.tor` ou `o D.tor`, \
regista obrigatoriamente `"o Doutor"` ou `"Doutor"` em `observacoes`, mesmo quando o `D` parece um `o` ligado.
Lê TODAS as palavras antes do primeiro marcador de parentesco (`fa.`/`f.ª`, `f.º`, `m.er`, `cr.`, etc.) \
ou estado civil (`S.ª`, `cas.`, `V.ª`, etc.) como parte do nome — \
NUNCA interrompas a leitura do nome ao encontrar uma palavra que parece um nome isolado. \
Exemplo: `"D. Anna M.a Leonor fa."` → `nome_original="D. Anna M.a Leonor"`, `parentesco="filha"`. \
**`do Ó`** é um apelido válido de devoção mariana (Nossa Senhora do Ó); após um nome próprio, \
faz parte do nome e NUNCA deve ser omitido. Exemplo: `"D. Maria do Ó"` → `nome_original="D. Maria do Ó"`. \
**`da Luz`/`daLuz`** é também um apelido de devoção (Nossa Senhora da Luz); após um nome próprio, \
é SEMPRE o apelido "da Luz" — **NUNCA é abreviatura de Luísa ou Luís**. \
Exemplo: `"Maria daLuz"` → `nome_original="Maria daLuz"`, `nome_expandido="Maria da Luz"` (criada). \
**`de Eça`/`d'Eça`** é um apelido nobre; pode ser mal lido como **`Beca`** ou **`de Beca`** \
porque o `E` cursivo parece `B` (confusão B↔E). Se leste `"Beca"` ou `"de Beca"` num apelido \
após prefixo honorífico `D.`/`D.ª`, reconsidera obrigatoriamente: é quase certamente `de Eça`.
**`de Alpoem`** é um apelido nobre/familiar encontrado nesta freguesia; pode ser mal lido como \
**`de Afonseca`** porque em cursivo `lp` parece `f`, `oe` parece `ons` e `m` parece `ca`. \
Se leste `"de Afonseca"` num fogo onde o contexto sugere outro apelido, reconsidera: \
pode ser `"de Alpoem"`. Todos os membros do mesmo fogo partilham o apelido do cabeça.
**`Carv.o` / `Carv.` / `Car.o`** = apelido **Carvalho** (`de Carvalho` / `do Carvalho`); \
pode ser mal lido como `Lano`, `Lan.`, `Lam.` ou `Lão` porque em cursivo `r`+`v` juntos \
parecem `n` (as hastes fundem-se). Se leste `"de Lano"` ou `"de Lan."` e esse topónimo \
não existe na freguesia, reconsidera: é quase certamente `Carv.o` = **de Carvalho**.
**`S. Paio` / `de S. Paio`** é um apelido de devoção (Santo Paio — São Pelayo); \
pode ser mal lido como `S. Cajo`, `S. Caio` ou `S. Cayo` porque o `P` cursivo maiúsculo \
assemelha-se a `C`. Se leste `"Cajo"`, `"Caio"` ou forma semelhante após `S.` (São/Santo) \
e o resultado não for um nome de santo ou apelido português reconhecível, \
reconsidera: quase certamente é `"Paio"` (Santo Paio = São Pelayo). \
Exemplo: `"Maria de S. Cajo"` → `nome_expandido="Maria de São Paio"`.
**`de S. do Carmo`**: nesta fórmula onomástica, `S.` significa **Senhora** (devoção mariana), \
não `São`. Portanto, `de S. do Carmo` deve ser expandido para **`de Senhora do Carmo`** \
e NUNCA para `de São do Carmo`.
**Atenção paleográfica — `Pace lo`**: o nome `Paulo` pode ser mal lido como `Pace lo` devido à forma como as letras cursivas se ligam. Se leres `Pace lo` (ou semelhante) como nome próprio, reconsidera: é quase certamente `Paulo`. O `nome_original` deve registar a forma que parece estar escrita se assim o entenderes, mas o `nome_expandido` DEVE ser `"Paulo"`. Se leres "Pace lo", nunca o convertas para "Marcelo" ou outro nome.
**Atenção paleográfica — `Ilena Iosefa` / `Ilena Josefa`**: o nome `Ilena` (variante arcaica de Helena) \
tem um `I` maiúsculo cursivo que é comumente mal lido como `M`, podendo transformar "Ilena" \
em "Maria". Além disso, o segundo nome `Iosefa` ou `Josefa` confunde-se facilmente com `Iozé` ou `José` (com a laçada inferior do f confundida com a de um z). \
Se leste a combinação `"Maria Iozé"` ou `"Maria José"`, reconsidera os traços iniciais: se as \
palavras originais parecerem ter a laçada inferior característica de Josefa/Iosefa e começar por I em vez de M (Ilena), o `nome_original` \
é **"Ilena Josefa"** (ou "Ilena Iosefa"), mas o `nome_expandido` DEVE ser obrigatoriamente **"Helena Josefa"**.
2. Expande as abreviaturas do NOME em `nome_expandido` usando a tabela fornecida. \
**Atenção a formas com determinante colado**: `daS.a` / `daS.º` → "da Silva"; \
`de A.` / `de Ar.º` → "de Araújo"; `doS.` → "do Sousa"; etc. \
O determinante (de/da/do/d') faz parte do apelido e **nunca deve ficar por expandir**. \
**Apelidos abreviados após o nome próprio**: se uma abreviatura imediatamente após o nome \
próprio designar um apelido (ex: `Pr.ª` / `Pr.a` / `P.ª` / `P.ra` → "Pereira"; \
`da S.a` / `da S.ª` → "da Silva"; \
`Sz.ª` → "Sousa"), ela faz parte do nome e DEVE ser expandida \
em `nome_expandido`. **Nunca confundas abreviaturas de apelido com abreviaturas de papel**: \
`Pr.ª` = Pereira (apelido), **não** é Prima nem Parentesco. \
**CRÍTICO — `S.ª`/`S.a` SEM `da`/`de` antes NUNCA é apelido "Silva"**: \
`S.ª` ou `S.a` isoladas (sem preposição imediatamente antes) são SEMPRE estado civil **solteira**. \
"Silva" como apelido exige OBRIGATORIAMENTE `da S.ª` / `da S.a` — o `da` tem de estar escrito no manuscrito. \
Exemplos: `"Tereza Barbara S.a"` → `nome_expandido="Teresa Bárbara"`, `estado_civil="solteira"` (NÃO Silva). \
`"Manoel da S.ª"` → `nome_expandido="Manuel da Silva"`, estado civil pelo contexto.
**CRÍTICO — nunca inventes expansão para apelido abreviado incerto**: se o apelido abreviado \
não tiver correspondência inequívoca na tabela de siglas e a leitura não for segura (ex.: `Carra.a`), \
remove esse apelido de `nome_expandido` e mantém apenas a parte certa do nome. \
Exemplo: `"Maria Tereza Carra.a"` → `nome_expandido="Maria Teresa"` (sem expandir o apelido incerto).
**NUNCA acrescentes `da`/`de` a um apelido se não estiver escrito no manuscrito.**
3. A cabeça do fogo (`#`) tem parentesco "cabeça". O primeiro membro de um grupo `//` \
tem SEMPRE parentesco "cabeça" (é chefe do seu sub-agregado). \
**Se aparecer explicitamente um grau de parentesco relativo ao fogo `#` principal** \
(ex: "Sogro", "Sogra", "Genro", "Nora", "Cunhado", "Cunhada", `cund.`/`cunh.`/`cund.ª`/`cunh.ª`, "Tio", "Avô", etc.), \
coloca esse termo expandido em `parentesco` (não em `observacoes`). \
A regra anterior aplica-se APENAS quando a pessoa é a cabeça de um sub-fogo `//`; \
nesse caso mantém `parentesco="cabeça"` e coloca o grau em `observacoes`. \
Se o termo (ex: "sogra") aparecer num membro do MESMO fogo (sem `//` na linha), \
usa directamente `parentesco="sogra"` (ou equivalente) e NÃO repitas em `observacoes`. \
A observação "sub-fogo do fogo N" é gerada automaticamente pelo sistema. \
**CRÍTICO — `cunh.`/`cund.` indica SEMPRE sub-fogo (`//`)**: `cunh.`/`cund.`/`cunh.ª`/`cund.ª` \
(cunhado/cunhada) junto a um nome em início de linha pressupõe OBRIGATORIAMENTE um marcador `//` \
antes — mesmo que não o tenhas lido claramente. Se vires `cunh.`/`cund.` após um nome, \
trata como sub-fogo: `simbolo="//"`, `parentesco="cabeça"`, `observacoes` inclui "cunhado" / "cunhada". \
NUNCA atribuas `parentesco="irmão"` a uma pessoa com `cunh.`/`cund.` — cunhado ≠ irmão. \
**Estado civil com `cunh.`**: `V.º`/`V.ª` que apareça ANTES de `cunh.`/`cund.` é estado civil \
**viúvo/viúva** e NUNCA deve ser ignorado ou substituído por "solteiro". `cunh.`/`cund.` não \
afecta nem altera o estado civil — vai apenas para `observacoes`. \
**Exemplo concreto**: `11 Rodrigo de Freitas V.º cunh.` → \
`simbolo="//"`, `nome_original="Rodrigo de Freitas"`, `parentesco="cabeça"`, \
`estado_civil="viúvo"` (V.º lido antes de cunh.), `observacoes="cunhado"`.
4. `f.al` após um nome **masculino** (sexo=M) = o membro é um trabalhador do fogo ("oficial"); \
usa `parentesco="oficial"` (não coloques nada em `observacoes`). \
`f.al` após um nome **feminino** (sexo=F) = leitura errada de `f.ª` → `parentesco="filha"`, \
nada em `observacoes`. \
**CRÍTICO — NUNCA inferir "oficial" por aproximação**: só usa `parentesco="oficial"` quando leres \
explicitamente `f.al` (ou variante paleográfica inequívoca da mesma sigla). \
**Cargos profissionais (Aprendiz, Companheiro, Mestre, Caixeiro)**: Devem ser SEMPRE colocados na coluna \
`parentesco` (como "aprendiz", "companheiro", "mestre", "caixeiro") e NÃO em `observacoes`. \
`Comp.o`/`Comp.`/`Companhr.o` = **companheiro** → `parentesco="companheiro"`. \
`Caixr.o`/`Caixr`/`Caxr.o` = **caixeiro** → `parentesco="caixeiro"`. \
`Mestre` = **mestre** → `parentesco="mestre"`. \
`Ap.`/`Apdz.`/`Aprendiz` = **aprendiz** → `parentesco="aprendiz"`.
5. `sp.` / `Sep.` / `sep.` / `Se.p` / `se.p` após um nome = a pessoa está separada → \
coloca **"separado"** (se `sexo` = M) ou **"separada"** (se `sexo` = F) em `observacoes`; \
o `estado_civil` mantém-se normalmente "casado/a".
6. **Ausente ou Preso** (`aus.te` / `auz.te` / `ausente` / `abz.` / `obz.` / `abz.te`): \
   - `aus.te` ou `auz.te` / `abz.te` no final de uma linha = a própria pessoa está ausente → coloca `"ausente"` em `observacoes`. \
   - `hom. abz.` / `hom. obz.` após o nome de uma mulher = o marido está ausente → coloca `"homem ausente"` em `observacoes`. \
   - `hom. prezo` / `hom. prez.` após o nome de uma mulher = o marido está preso → coloca `"homem preso"` em `observacoes`. \
   - `m.er abz.` / `m.er obz.` / `m.er abz.te` / `m.er aus.te` / `cas. m.er abz.te` após o nome de um homem = a mulher está ausente → coloca `"mulher ausente"` em `observacoes`. \
   Nos casos de cônjuge (hom. ou m.er), não assinales como solteiro/a, o `estado_civil` tem de ser OBRIGATORIAMENTE "casado" / "casada", porque se a mulher ou marido estão ausentes, eles estão casados.
6. **Estado civil por omissão**: se o estado civil não estiver explicitamente indicado \
no manuscrito, aplica esta regra:
   - Parentesco **filho, filha, irmão, irmã, criado, criada, servo, serva, neto, neta, sobrinho, sobrinha, afilhado, afilhada** \
→ assume `"solteiro"` ou `"solteira"` conforme o sexo.
   - Se a **cabeça masculina** for seguida de uma pessoa com parentesco **"mulher"** no mesmo fogo, \
ambos têm `estado_civil` **"casado"** e **"casada"** respectivamente, **mas APENAS se a cabeça \
não tiver marcador explícito de viuvez** (`V.o`/`V.ª`/`V.a`/`V.º`/`V.°`/`viu.`/`viu.o`/`viuva`). \
**EXCEPÇÃO CRÍTICA**: se a cabeça tiver `V.o` ou qualquer variante de viúvo/viúva explicitamente \
escrita no manuscrito, o seu `estado_civil` mantém-se **"viúvo"** (é uma segunda união/casamento) — \
a mulher fica com `estado_civil="casada"`. Nunca uses "casado" para anular um `V.o` explícito.
   - **Cabeça** sem "mulher" no mesmo fogo, e sem indicação explícita → usa `"desconhecido"`.
   - **REGRA DE DESAMBIGUAÇÃO — cabeça sozinha**: para uma cabeça sem "mulher" no fogo, \
aplica esta hierarquia estrita: \
     1. Se identificaste explicitamente **solteiro/solteira** (`S.º`/`soltr.`/`solt.`) → usa `"solteiro/a"`. \
     2. Se identificaste explicitamente **viúvo/viúva** (`V.ª`/`V.o`/`viu.`/etc.) → usa `"viúvo/a"`. \
     3. Se identificaste `cas.`/`caz.` explícito OU cônjuge ausente/preso (`hom. abz.`/`hom. prezo`/`m.er abz.`/`m.er abz.te`) → usa `"casado/a"`. \
     4. **Em QUALQUER outro caso** (incluindo quando serias tentado a escrever "casado/a" \
sem marcador explícito) → usa **`"viúvo"`** (se sexo M) ou **`"viúva"`** (se sexo F). \
Justificação: o modelo confunde `V.ª` (viúva) com abreviaturas de casado/a, mas raramente \
confunde `S.ª`/`S.º` (solteiro/a). Se não é solteiro/a e não há marcador explícito de \
casamento → é quase certamente viúvo/a mal lido. "Desconhecido" não deve ser usado para cabeças.
7. **Padre (`P.e` / `o P.e`)**: o artigo "o" antes de `P.e` (ex: `o P.e Thomaz`) \
é simplesmente "o Padre" — trata da mesma forma que `P.e` isolado; "o" não faz parte do nome. \
Distingue dois casos:
   - `P.e` **antes do nome da cabeça** (`#` ou `//`): a cabeça é um padre → \
`parentesco = "cabeça"`, `estado_civil = "solteiro"`, regista `"Padre"` em `observacoes`. \
**`P.e` NUNCA entra em `nome_original` nem `nome_expandido` — mas OBRIGATORIAMENTE aparece \
em `observacoes` como `"Padre"`. Nunca o omitas.** \
Exemplo: `// o P.e Thomaz Fran.co da S.a [linha] D.M.` → `nome_original="Thomaz Fran.co da S.a"`, \
`nome_expandido="Tomás Francisco da Silva"`, `parentesco="cabeça"`, `estado_civil="solteiro"`, \
`confessou="sim"`, `observacoes="Padre, D.M."`. \
Os outros membros do fogo **sem indicação explícita de parentesco** assumem `"irmão"` / `"irmã"`. \
Se o parentesco estiver **explicitamente escrito** no manuscrito (ex: "Primo", "Sobrinho", "Mãe", "Pai"), \
usa esse valor — **nunca substituas parentesco explícito pela assunção de irmão/irmã**. \
**CRÍTICO — parentesco "mãe"**: a palavra aparece no manuscrito como `Mãe`, `mãi`, `May`, `Mai`, `Mae`, `Maj` \
(leitura paleográfica de 'ã' como 'ay'/'ai'/'ae'/'aj') — TODAS estas formas significam parentesco `"mãe"`. \
O estado civil acompanhante é sempre o que estiver escrito (ex: `V.a` = `"viúva"`). \
Exemplo: `Maria Tereza V.a May` → `parentesco="mãe"`, `estado_civil="viúva"`.
**DISTINÇÃO CRÍTICA — `m.er`/`mul.` (mulher/esposa) vs formas de "mãe"**: \
`m.er` e `mul.` SEMPRE produzem `parentesco="mulher"` (esposa) — **NUNCA "mãe"**. \
Os indicadores de "mãe" (`Mãe`, `mãi`, `May`, `Mai`, `Mae`, `Maj`) são palavras \
escritas por extenso (total ou foneticamente) e NUNCA se confundem com a abreviatura \
`m.er`/`mul.` que tem as letras `e`+`r` explicitamente escritas após o `m`. \
**Se vires `m.er` ou `mul.` após o nome de uma mulher, é SEMPRE a esposa**, \
`parentesco="mulher"`, independentemente do fogo ou do contexto. \
Exemplos de distinção: \
`"Patornilla m.er"` → `parentesco="mulher"` (esposa, NÃO mãe); \
`"Roza Maria m.er"` → `parentesco="mulher"` (esposa, NÃO mãe); \
`"Maria Tereza V.a May"` → `parentesco="mãe"` (a palavra May = Mãe, escrita por extenso).
   - `P.e` **antes do nome de um membro não-cabeça**: esse membro é um padre → regista \
`"Padre"` em `observacoes`, `estado_civil = "solteiro"`. O parentesco para com a cabeça \
usa o que estiver indicado no manuscrito; se não houver indicação, usa `"outro"`.
7c. **`D.M.` como indicador de padre** — se `D.M.` aparecer na posição de confissão \
(após a linha horizontal), a pessoa é quase certamente um padre, mesmo que `P.e` não tenha \
sido identificado antes do nome. Aplica estas correcções: \
   - Acrescenta `"Padre"` a `observacoes` (para além de `"D.M."` que já lá fica). \
   - Define `estado_civil = "solteiro"`. \
   - `confessou = "sim"` (como qualquer outra palavra nessa posição). \
**EXCEPÇÃO IMPORTANTE**: Se a pessoa for identificada com a palavra "Reverendo" (`R.do` / `Rev.do`), **NÃO acrescentes** `"Padre"` às observações por causa do `D.M.` — nesses casos, mantém a pessoa Apenas como Reverendo (não Padre). \
**Atenção paleográfica**: `o P.e` antes do nome é frequentemente lido como `D.` (Dom) — \
se a abreviatura inicial antes do nome parece `D.` mas a pessoa tem `D.M.` na posição de \
confissão, é muito provável que o `D.` seja `o P.e` mal lido. Nesse caso, `"Dom"` / `"D."` \
NÃO entra no nome; o nome começa na palavra a seguir.
7d. **`R.do` / `Rev.do` (Reverendo)**: quando `R.do` ou `Rev.do` aparecer antes de um nome \
(quer em cabeça `#` quer em sub-fogo `//`, quer em membro não-cabeça), coloca `"Reverendo"` \
em `observacoes` — **nunca omitas este marcador**. Não faz parte do `nome_original` nem do \
`nome_expandido`. **Se a mesma pessoa tiver também `P.e`, mantém apenas `"Reverendo"`** \
(não acrescentes `"Padre"`, para evitar duplicação hierárquica do título eclesiástico). \
Se tiver a marca da confissão `D.M.`, a pessoa já está identificada como `Reverendo`, portanto não deve ser inferida como "Padre". \
Exemplo: `11. R.do Iozé P.e` → `nome_original="Iozé"`, `observacoes="Reverendo"`.
7h. **`Arcip.` / `Arcipreste` / `Arcipr.e` (Arcipreste)**: quando esta sigla ou palavra \
aparecer antes do nome, coloca `"Arcipreste"` em `observacoes` — não entra no nome. \
Se a pessoa já estiver marcada como `"Reverendo"`, mantém ambos: `"Reverendo, Arcipreste"`. \
`Arcip.` nunca implica `"Padre"` adicional.
7e. **`L.do` / `oL.do` / `Lic.do` (Licenciado)**: quando `L.do`, `oL.do` ou `Lic.do` aparecer \
antes de um nome, coloca `"Licenciado"` em `observacoes` — **nunca omitas este marcador**. \
Não faz parte do `nome_original` nem do `nome_expandido`. O artigo `o` antes de `L.do` \
(ex: `o L.do Luis Manoel`) é simplesmente "o Licenciado" — trata da mesma forma. \
Exemplo: `# oL.do Luis Manoel de Alpoem` → `nome_original="Luis Manoel de Alpoem"`, `observacoes="Licenciado"`.
7ea. **`D.or` / `Dr.` / `oD.or` / `Dor` (Doutor)**: quando estas formas aparecerem antes de um nome, \
coloca `"Doutor"` em `observacoes` — **nunca omitas este marcador**. \
Não faz parte do `nome_original` nem do `nome_expandido`. O artigo `o` em `oD.or` é apenas \
o artigo ("o Doutor"), não entra no nome. \
Exemplo: `# oD.or Bento Iozé de Moura` → `nome_original="Bento Iozé de Moura"`, \
`nome_expandido="Bento José de Moura"`, `observacoes="Doutor"`.
7eb. **`oManjor` / `o Major` / `Manjor` / `Major` (Major)**: quando estas formas aparecerem \
antes do nome, trata-se do título **Major** (cargo militar), NÃO de Doutor e NÃO de nome próprio. \
Remove obrigatoriamente o título do `nome_original` e `nome_expandido` e coloca `"Major"` \
em `observacoes`. \
**Regra anti-erro obrigatória**: `oManjor`/`Manjor` NUNCA pode ser expandido para `"Doutor"` \
nem para `"Manoel"` no nome.
7f. **`Conego` / `Cónego` / `Can.go` / `Con.go` / `Can.º` (Cónego — Cânone)**: quando esta \
palavra ou abreviatura aparecer antes de um nome, coloca `"Cónego"` em `observacoes` — \
**NUNCA a incluas em `nome_original` nem `nome_expandido`**, nem como parte do nome próprio. \
O parentesco e demais campos seguem o manuscrito normalmente — em especial, se aparecer `f.` \
após o nome, `parentesco = "filho"` (o marcador de parentesco NÃO é anulado pelo título). \
Se a mesma pessoa tiver também `R.do`/`P.e`, acumula em `observacoes`: ex. `"Reverendo, Cónego"`. \
Exemplo: `o R. Conego Paulo de Carv.o f.` → `nome_original="Paulo de Carv.o"`, \
`nome_expandido="Paulo de Carvalho"`, `parentesco="filho"`, `observacoes="Reverendo, Cónego"`.
7g. **`Escud.` / `Escud.ro` / `Escudeiro` (Escudeiro — Esquire)**: quando `Escud.`, `Escud.ro` \
ou `Escudeiro` aparecer após ou antes do nome de uma pessoa, coloca `"Escudeiro"` em `observacoes` \
— **nunca omitas este marcador**. Não faz parte do `nome_original` nem do `nome_expandido`. \
O parentesco e demais campos seguem o manuscrito normalmente (ex: `cr.` após o nome = `parentesco="criado"`). \
Pode coexistir com outros títulos em `observacoes`. \
Exemplo: `Teodozio Gomes Escud.ro cr.` → `nome_original="Teodozio Gomes"`, \
`parentesco="criado"`, `estado_civil="solteiro"`, `observacoes="Escudeiro"`.
7b. **Proibição de irmão/irmã por omissão**: NUNCA atribuas `"irmão"` ou `"irmã"` a uma \
pessoa cujo único indicador seja o estado civil (`S.ª`, `V.ª`, `cas.`, etc.) sem qualquer \
marcador de parentesco explícito (`Ir.`, `f.º`, etc.) e sem marcador `//`. \
Se o parentesco for genuinamente desconhecido, usa `"desconhecido"`, não `"irmão/irmã"`.
8. **`f.º m.` / `f.ª m.` e `m.` isolado**: `m.` (em qualquer posição após o nome) significa \
**"menor"** (criança isenta de confissão) — coloca `"menor"` em `observacoes`. \
CUIDADO: a sigla `"aus.te"` ou `"auz.te"` significa `"ausente"`, não confundas com `"m."` (menor). \
Se precedido de `f.º`/`f.ª`, o parentesco fica `"filho"`/`"filha"`; \
se `m.` aparecer isolado sem indicação de parentesco, determina o parentesco pelo contexto normal. \
**`m.` NUNCA significa "mãe"** — se o manuscrito referir a mãe, estará escrito por extenso como `Mãe`, `mãi`, `May`, `Mai`, `Mae` ou `Maj` (ver Regra 7). \
Exemplo concreto: `"Francisco Jozé f.º m."` → `nome_original="Francisco Jozé"`, \
`parentesco="filho"`, `observacoes="menor"` — o `m.` no final NUNCA deve ser ignorado.
8b. **Enjeitado/a** (`enjeitado`, `enjeitada`, `Ing.`, `Ingeitado`, `Ingeitada` ou variantes): \
criança abandonada acolhida no fogo. `parentesco` = `"outro"`; \
coloca `"enjeitado"` (sexo M) ou `"enjeitada"` (sexo F) em `observacoes`. \
Estes termos **NUNCA** fazem parte de `nome_original` ou `nome_expandido`; remove-os sempre do nome. \
Exemplo: `"Florinda Ingeitada m."` → `parentesco="outro"`, `observacoes="enjeitada, menor"`.
8c. **Escudeiro** (`Escudr.o` / `Escudr.º` / `Ecudr.º` / `Escudro` / `Escudiero`): \
título/função de criado nobre → coloca "Escudeiro" em `observacoes`; \
`parentesco` mantém-se **"criado"**.
8d. **In minoribus** (`inmino.` / `in mino.` / `in min.`): a pessoa recebeu ordens menores \
mas não foi ordenada sacerdote → coloca `"in minoribus"` em `observacoes`; \
não afecta `parentesco` nem `estado_civil`.
8e. **Capitão** (`Capp.am` / `oCapp.am`): título militar → coloca `"Capitão"` em `observacoes`; \
não afecta `parentesco` nem `estado_civil`.
8f. **Primeira tença** (`pr.a Ten.ca` / `prim.a Ten.ca` / `pr.a Tença`): renda ou pensão anual \
→ coloca `"primeira tença"` em `observacoes`; não afecta `parentesco` nem `estado_civil`.
9. **`F.` como anotação de falecimento** (acima ou ao longo da linha horizontal): a pessoa \
**faleceu** — coloca **"falecida"** (sexo F) ou **"falecido"** (sexo M) em `observacoes`. \
`F.` é independente de "Confes.": `confessou` segue a regra habitual mesmo que `F.` esteja presente. \
**`F.` NÃO altera o `estado_civil`** — mantém o que estiver explicitamente escrito \
antes da linha (S.ª / solt. = solteira, cas. = casada, viu. = viúva, etc.). \
Se não houver indicação de estado civil, aplica a regra de omissão normal; \
**nunca derives "viúvo/a" da simples presença de `F.`**.
10. Se não conseguires ler uma palavra, usa "[ilegível]".
10b. **Página em branco**: se a imagem estiver em branco (sem nomes/fogos), ou se apenas tiver
uma anotação de controlo como `Branca`, `Em branco`, `Página em branco`, devolve **exatamente**:
`{{"lugar_atual": null, "lugar_proximo": null, "grupos": []}}`.
Não inventes pessoas, lugares, fogos nem observações a partir desta anotação.
11. **Lugar**: um indicador de lugar (texto em letras maiores) aplica-se APENAS aos fogos \
que aparecem A SEGUIR a esse indicador, nunca aos fogos anteriores. \
Há três posições possíveis para o indicador: \
a) **Antes do primeiro fogo** da página → usa `lugar_atual` (nível da página). \
b) **A meio da página, entre fogos** → coloca o nome em `lugar_novo` do **primeiro fogo** \
que se segue ao indicador; os fogos anteriores a esse indicador pertencem ao lugar anterior. \
c) **Depois do ÚLTIMO fogo** da página (fim de página, a anunciar a secção seguinte) → usa \
`lugar_proximo`; os fogos desta página pertencem ao lugar anterior, não a este novo. \
`lugar_novo` deve ser `null` em todos os fogos onde o lugar não muda. \
`lugar_proximo` deve ser `null` se não existir indicador de lugar no fim da página."""

# ====================================================================
# CHAMADA À API GEMINI
# ====================================================================
class QuotaDiariaEsgotada(Exception):
    """Excepção levantada quando o limite diário da API é atingido."""
    pass


def _extrair_delay_retry(exc_str: str) -> int:
    """Extrai o número de segundos de espera sugerido pelo erro 429."""
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", exc_str)
    if m:
        return int(m.group(1)) + 5   # margem de segurança
    m = re.search(r"retry in\s+([\d.]+)s", exc_str)
    if m:
        return int(float(m.group(1))) + 5
    return RETRY_DELAY


def call_gemini(image: Image.Image, prompt: str, model, lugar_atual: str = "") -> dict:
    """Envia imagem + prompt ao Gemini e devolve o dict parseado.
    Se lugar_atual for fornecido, é adicionado como contexto da página para
    que o modelo saiba qual o lugar em vigor e só devolva um novo se mudar.
    Levanta QuotaDiariaEsgotada se o limite diário for atingido.
    """
    # Contexto de lugar passado por página para dar "memória" ao modelo
    if lugar_atual:
        page_context = (
            f"\n\n**CONTEXTO DESTA PÁGINA**: O lugar actualmente em vigor é "
            f"'{lugar_atual}'. Devolve `null` em `lugar_atual` se NÃO aparecer "
            f"um lugar/rua diferente ANTES do primeiro fogo desta página. "
            f"Se aparecer um indicador de lugar no FINAL da página (depois do último fogo), "
            f"coloca-o em `lugar_proximo`."
        )
    else:
        page_context = (
            "\n\n**CONTEXTO DESTA PÁGINA**: Ainda não foi identificado nenhum lugar. "
            "Se aparecer um nome de lugar ou freguesia ANTES do primeiro fogo, regista-o em `lugar_atual`. "
            "Se aparecer depois do último fogo, regista-o em `lugar_proximo`."
        )
    full_prompt = prompt + page_context

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                [full_prompt, image],
                generation_config={"temperature": 0.1, "max_output_tokens": 8192},
            )
            text = response.text.strip()

            # Remove eventual envolvimento em bloco de código markdown
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

            return json.loads(text)

        except json.JSONDecodeError as exc:
            # Em páginas sem conteúdo genealógico o modelo pode devolver apenas
            # uma frase do tipo "página em branco" em vez de JSON.
            if re.search(r"(?i)\b(branca|em branco|p[aá]gina em branco)\b", text):
                return {"lugar_atual": None, "lugar_proximo": None, "grupos": []}

            print(f"    [tentativa {attempt}/{MAX_RETRIES}] Resposta não é JSON válido: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print("    AVISO: a ignorar esta página (resposta inválida).")
                return {"lugar_atual": None, "lugar_proximo": None, "grupos": []}

        except Exception as exc:
            exc_str = str(exc)
            is_429  = "429" in exc_str or "quota" in exc_str.lower() or "rate" in exc_str.lower()

            if is_429:
                wait = _extrair_delay_retry(exc_str)
                # Quota diária esgotada: delay > 5 minutos → não vale a pena esperar
                if wait > 300:
                    raise QuotaDiariaEsgotada(
                        f"Quota diária atingida. Retoma amanhã ou aumenta o plano.\n  Detalhe: {exc}"
                    )
                print(f"    [tentativa {attempt}/{MAX_RETRIES}] Rate limit 429 — "
                      f"a aguardar {wait}s antes de tentar novamente...")
                time.sleep(wait)
            else:
                print(f"    [tentativa {attempt}/{MAX_RETRIES}] Erro na API: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    print("    AVISO: a ignorar esta página (erro na API).")
                    return {"lugar_atual": None, "lugar_proximo": None, "grupos": []}

    return {"lugar_atual": None, "lugar_proximo": None, "grupos": []}

# ====================================================================
# CONSTRUÇÃO DA TABELA FINAL
# ====================================================================
def build_table(pages_data: list[dict]) -> list[dict]:
    """
    Percorre os dados de todas as páginas e constrói a tabela final,
    mantendo estado incremental: Id, Fogo, Lugar e associação // → #.
    """
    rows = []
    current_id    = 1
    current_fogo  = 0
    current_lugar = ""
    last_hash_fogo = None   # número de fogo do último # encontrado
    fogo_parent: dict[int, int] = {}  # sub-fogo → fogo # pai (para chain resolution)
    _f_al_re = re.compile(r"(?<!\w)f\.?\s*al\.?(?!\w)", re.IGNORECASE)

    for page_data in pages_data:
        # Guardar o lugar em vigor antes desta página, para os grupos ~ de continuação.
        # Grupos ~ pertencem a um fogo que COMEÇOU na página anterior e herdam esse lugar;
        # o novo lugar_atual desta página aplica-se apenas aos fogos que COMEÇAM aqui.
        lugar_antes_desta_pagina = current_lugar

        # Actualiza o lugar se a página contiver um novo
        novo_lugar = page_data.get("lugar_atual")
        if novo_lugar:
            current_lugar = novo_lugar.strip()

        for grupo in page_data.get("grupos", []):
            # Lugar novo indicado a meio da página, antes deste fogo específico
            lugar_novo_grupo = grupo.get("lugar_novo")
            if lugar_novo_grupo:
                lugar_novo_grupo = lugar_novo_grupo.strip()
                # Se o modelo devolveu qualquer variante de "Doutra Parte"
                # (incluindo nomes-base errados ou formas truncadas como "Doutra Par"),
                # reconstruímos sempre a partir do current_lugar correcto.
                if re.search(r"doutra\s+par", lugar_novo_grupo, re.IGNORECASE):
                    current_lugar = current_lugar + " - Doutra Parte"
                else:
                    current_lugar = lugar_novo_grupo

            simbolo = grupo.get("simbolo", "#").strip()
            # Normalizar variantes do marcador de sub-fogo → "//"
            # O OCR lê frequentemente // como "11.", "11", "ll", "ll.", "‖"
            if re.match(r'^(11\.?|ll\.?|\|\|\.?|‖)$', simbolo):
                simbolo = "//"
            # "~" = continuação da página anterior: não abre novo fogo
            if simbolo != "~" or current_fogo == 0:
                current_fogo += 1

            # Grupos ~ herdam o lugar da página anterior (o fogo começou lá);
            # todos os outros grupos usam o lugar actualizado desta página.
            lugar_deste_fogo = lugar_antes_desta_pagina if simbolo == "~" else current_lugar

            obs_de_fogo = ""
            if simbolo == "#":
                last_hash_fogo = current_fogo
            elif simbolo in ("//", "ll", "‖"):
                # last_hash_fogo != current_fogo é sempre True (// incrementou current_fogo)
                if last_hash_fogo is not None:
                    obs_de_fogo = f"sub-fogo do fogo {last_hash_fogo}"
                    fogo_parent[current_fogo] = last_hash_fogo
            # simbolo "~": continuação sem obs extra nem novo fogo

            for pessoa in grupo.get("pessoas", []):
                obs_pessoa  = (pessoa.get("observacoes") or "").strip()
                parentesco  = (pessoa.get("parentesco") or "").strip()
                sexo        = (pessoa.get("sexo") or "").strip()
                nome_original = (pessoa.get("nome_original") or "").strip()
                nome_expandido = (pessoa.get("nome_expandido") or "").strip()
                tem_f_al = bool(_f_al_re.search(" ".join([nome_original, nome_expandido, obs_pessoa])))
                tem_oficial_explicito = bool(re.search(r"(?i)\bo[fe]c?ç?ial\b", " ".join([nome_original, nome_expandido])))

                # Lida com o falso positivo "f.al" (oficial) em mulheres, que costuma ser leitura errada de "f.ª" (filha)
                if parentesco == "oficial" and sexo == "F" and tem_f_al and not tem_oficial_explicito:
                    parentesco = "filha"
                # Apenas removemos "oficial" de parentesco se for um falso positivo originado de "f.al" num homem?
                # A pedido do utilizador, "oficial" deve manter-se como parentesco quando explicitamente escrito.
                elif parentesco == "oficial" and tem_f_al and not tem_oficial_explicito:
                    parentesco = "outro"
                    if "oficial" not in obs_pessoa.lower():    
                        obs_pessoa = "; ".join(filter(None, ["oficial", obs_pessoa]))
                elif "oficial" in obs_pessoa.lower() and not (tem_f_al or tem_oficial_explicito):
                    # Evita "oficial" sem evidência textual explícita.
                    obs_pessoa = re.sub(r"\s*;\s*;\s*", "; ", obs_pessoa)
                    obs_pessoa = re.sub(r"^\s*;\s*|\s*;\s*$", "", obs_pessoa).strip()

                obs_total  = "; ".join(filter(None, [obs_de_fogo, obs_pessoa]))

                rows.append({
                    "Id":             current_id,
                    "Fogo":           current_fogo,
                    "Lugar":          lugar_deste_fogo,
                    "NomeOriginal":   nome_original,
                    "NomeAtualizado": nome_expandido,
                    "Sexo":           sexo,
                    "EstadoCivil":    (pessoa.get("estado_civil") or "").strip(),
                    "Parentesco":     parentesco,
                    "Confessou":      (pessoa.get("confessou") or "").strip(),
                    "Observações":    obs_total,
                })
                current_id += 1

        # Indicador de lugar no fim da página (após o último fogo) → aplica-se à PRÓXIMA página
        lugar_proximo = page_data.get("lugar_proximo")
        if lugar_proximo:
            current_lugar = lugar_proximo.strip()

    # ------------------------------------------------------------------
    # Pós-processamento: normalizar campo Confessou
    # Qualquer sigla/texto no campo de confissão implica confessou="sim".
    # Ex.: "Sopti" (variante paleográfica) deve contar como confissão.
    # ------------------------------------------------------------------
    _conf_tokens_pat = re.compile(r"^(?:conf(?:es)?\.?|sopti\.?|spti\.?)$", re.IGNORECASE)
    for r in rows:
        conf_raw = (r["Confessou"] or "").strip()
        conf_norm = conf_raw.lower()
        if conf_norm in ("sim", "não", "nao", "ilegível", "ilegivel", ""):
            continue
        r["Confessou"] = "sim"
        if conf_raw and conf_raw.lower() not in r["Observações"].lower():
            if not _conf_tokens_pat.match(conf_raw):
                r["Observações"] = "; ".join(filter(None, [r["Observações"], conf_raw]))
            elif re.search(r"(?i)\bsopti\b|\bspti\b", conf_raw):
                r["Observações"] = "; ".join(filter(None, [r["Observações"], conf_raw]))

    # ------------------------------------------------------------------
    # Pós-processamento: corrigir "casado/a" indevido em cabeças sozinhas
    # O modelo confunde V.ª (viúva) com abreviaturas de casamento.
    # Regra: cabeça sem "mulher" no fogo + sem cônjuge ausente → viúvo/a
    # ------------------------------------------------------------------
    focos_com_mulher: set[int] = {
        r["Fogo"] for r in rows if r["Parentesco"] == "mulher"
    }
    focos_com_ausente: set[int] = {
        r["Fogo"] for r in rows
        if "ausente" in r["Observações"].lower()
        or "preso" in r["Observações"].lower()
        or "separado" in r["Observações"].lower()
        or "separada" in r["Observações"].lower()
    }
    for r in rows:
        if (
            r["Parentesco"] == "cabeça"
            and r["EstadoCivil"] in ("casado", "casada")
            and r["Fogo"] not in focos_com_mulher
            and r["Fogo"] not in focos_com_ausente
        ):
            r["EstadoCivil"] = "viúva" if r["Sexo"] == "F" else "viúvo"

    # ------------------------------------------------------------------
    # Pós-processamento: Vr.a/Vr.ª (Vieira) lido como viúvo no nome/estado
    # Quando o cabeça tem "mulher" no mesmo fogo, não pode ficar viúvo por
    # confusão de apelido. Corrige "Viúvo/Viúva" no nome para "Vieira".
    # ------------------------------------------------------------------
    _vr_sigla_orig_pat = re.compile(r"\bVr\.?\s*[aªoº]\b", re.IGNORECASE)
    _viuvo_nome_pat = re.compile(r"\bVi[uú]v[oa]\b", re.IGNORECASE)
    for r in rows:
        has_vr_sigla = bool(_vr_sigla_orig_pat.search(r["NomeOriginal"]))
        has_viuvo_no_nome = bool(_viuvo_nome_pat.search(r["NomeAtualizado"]))
        if has_vr_sigla and has_viuvo_no_nome:
            r["NomeAtualizado"] = _viuvo_nome_pat.sub("Vieira", r["NomeAtualizado"])

        if (
            r["Parentesco"] == "cabeça"
            and r["Fogo"] in focos_com_mulher
            and r["EstadoCivil"] in ("viúvo", "viúva")
            and has_viuvo_no_nome
        ):
            r["NomeAtualizado"] = _viuvo_nome_pat.sub("Vieira", r["NomeAtualizado"])
            r["EstadoCivil"] = "casada" if r["Sexo"] == "F" else "casado"

    # ------------------------------------------------------------------
    # Pós-processamento: D.M. na posição de confissão → Padre
    # D.M. é marcador consistente de padre; o modelo frequentemente lê
    # "o P.e" como "D." (Dom/Dona) e não identifica a pessoa como padre.
    # A detecção usa regex para cobrir variantes (D.M., d.m., D. M., DM).
    # EXCEPÇÃO: Se a pessoa já está identificada como Reverendo, não adiciona "Padre".
    # ------------------------------------------------------------------
    _dm_pat = re.compile(r'\bD\.?\s*M\.?', re.IGNORECASE)
    _dona_prefix = re.compile(r'^(Dom|Dona|D\.)\s+', re.IGNORECASE)
    for r in rows:
        obs = r["Observações"]
        if _dm_pat.search(obs) and "Padre" not in obs and "Reverendo" not in obs:
            r["Observações"] = "Padre; " + obs
            r["EstadoCivil"] = "solteiro"
            r["Sexo"] = "M"   # padres são sempre do sexo masculino
            # remover prefixo "Dom"/"Dona"/"D." do nome (leitura errada de "o P.e")
            r["NomeOriginal"]   = _dona_prefix.sub("", r["NomeOriginal"]).strip()
            r["NomeAtualizado"] = _dona_prefix.sub("", r["NomeAtualizado"]).strip()
            # corrigir parentesco feminino → masculino
            _fem_to_masc = {
                "irmã": "irmão", "criada": "criado", "filha": "filho",
                "neta": "neto", "sobrinha": "sobrinho", "serva": "servo",
                "prima": "primo",
            }
            r["Parentesco"] = _fem_to_masc.get(r["Parentesco"], r["Parentesco"])

    # ------------------------------------------------------------------
    # Pós-processamento: Reverendo e Arcipreste
    # - Reverendo prevalece sobre Padre (não manter ambos na mesma pessoa)
    # - Arcip./Arcipreste é título e nunca faz parte do nome
    # ------------------------------------------------------------------
    _arcip_prefix_orig = re.compile(r'^(?:o\s*)?Arcip(?:\.|reste|r\.e)\s+', re.IGNORECASE)
    _arcip_prefix_expd = re.compile(r'^(?:o\s*)?Arcipreste\s+', re.IGNORECASE)
    _arcip_obs = re.compile(r'\bArcip(?:\.|reste|r\.e)\b', re.IGNORECASE)
    for r in rows:
        had_arcip = (
            bool(_arcip_prefix_orig.search(r["NomeOriginal"]))
            or bool(_arcip_prefix_expd.search(r["NomeAtualizado"]))
            or bool(_arcip_obs.search(r["Observações"]))
        )

        if had_arcip:
            r["NomeOriginal"] = _arcip_prefix_orig.sub("", r["NomeOriginal"]).strip()
            r["NomeAtualizado"] = _arcip_prefix_expd.sub("", r["NomeAtualizado"]).strip()
            if "arcipreste" not in r["Observações"].lower():
                r["Observações"] = "; ".join(filter(None, [r["Observações"], "Arcipreste"]))
            r["Observações"] = _arcip_obs.sub("Arcipreste", r["Observações"])

        if "reverendo" in r["Observações"].lower() and "padre" in r["Observações"].lower():
            obs = re.sub(r'(?i)\bPadre\b', '', r["Observações"])
            obs = re.sub(r'\s*;\s*;\s*', '; ', obs)
            r["Observações"] = re.sub(r'^\s*;\s*|\s*;\s*$', '', obs).strip()

    # ------------------------------------------------------------------
    # Pós-processamento: Jacinto/Iacinto Ant.o Machado com Padre sem Sacristão
    # Caso recorrente em que a palavra Sacristão não é extraída.
    # ------------------------------------------------------------------
    _jacinto_machado_orig = re.compile(r'^(?:I|J)acint[oa]\s+Ant\.o\s+Machado$', re.IGNORECASE)
    _jacinto_machado_expd = re.compile(r'^Jacint[oa]\s+Ant[oó]nio\s+Machado$', re.IGNORECASE)
    for r in rows:
        if "padre" not in r["Observações"].lower() or "sacristão" in r["Observações"].lower():
            continue
        if not (_jacinto_machado_orig.match(r["NomeOriginal"]) or _jacinto_machado_expd.match(r["NomeAtualizado"])):
            continue
        # "Iacinto/Jacinto" neste padrão é leitura errada de "Sacristão".
        r["NomeOriginal"] = "Ant.o Machado"
        r["NomeAtualizado"] = "António Machado"
        r["Sexo"] = "M"
        r["EstadoCivil"] = "solteiro"
        r["Observações"] = "; ".join(filter(None, [r["Observações"], "Sacristão"]))

    # ------------------------------------------------------------------
    # Pós-processamento: "oP.e" lido como "D." + "Sopti" na confissão
    # Corrige entradas masculinas onde o prefixo "D." no nome é leitura
    # errada de "o P.e" (padre), sinalizada por Sopti/Spti.
    # ------------------------------------------------------------------
    _d_prefix = re.compile(r'^D\.\s+', re.IGNORECASE)
    for r in rows:
        if r["Sexo"] != "M":
            continue
        if "padre" in r["Observações"].lower() or "doutor" in r["Observações"].lower():
            continue
        if not _d_prefix.search(r["NomeOriginal"]):
            continue
        marker_text = " ".join([r["Confessou"], r["Observações"]])
        if not re.search(r"(?i)\bsopti\b|\bspti\b", marker_text):
            continue
        r["NomeOriginal"] = _d_prefix.sub("", r["NomeOriginal"]).strip()
        r["NomeAtualizado"] = _d_prefix.sub("", r["NomeAtualizado"]).strip()
        r["Observações"] = "; ".join(filter(None, ["Padre", r["Observações"]]))
        r["EstadoCivil"] = "solteiro"

    # ------------------------------------------------------------------
    # Pós-processamento: D.or/Dr./oD.or no nome → Doutor em observações
    # Títulos académicos não fazem parte do nome.
    # ------------------------------------------------------------------
    _doutor_prefix_orig = re.compile(r'^(?:o\s*)?(?:D\.?\s*or\.?|Doutor)\s+', re.IGNORECASE)
    _doutor_prefix_expd = re.compile(r'^(?:o\s*)?Doutor\s+', re.IGNORECASE)
    for r in rows:
        had_doutor_prefix = (
            bool(_doutor_prefix_orig.search(r["NomeOriginal"]))
            or bool(_doutor_prefix_expd.search(r["NomeAtualizado"]))
        )
        if not had_doutor_prefix:
            continue
        if "doutor" not in r["Observações"].lower():
            r["Observações"] = "; ".join(filter(None, ["Doutor", r["Observações"]]))
        r["NomeOriginal"] = _doutor_prefix_orig.sub("", r["NomeOriginal"]).strip()
        r["NomeAtualizado"] = _doutor_prefix_expd.sub("", r["NomeAtualizado"]).strip()

    # ------------------------------------------------------------------
    # Pós-processamento: oManjor mal lido como "Doutor Manoel"
    # Caso recorrente: "oManjor Jozé Frz." convertido indevidamente para
    # "Doutor Manoel Jozé Frz.". Corrige para título "Major" e remove
    # "Manoel" indevido do nome.
    # ------------------------------------------------------------------
    _manjor_false_doutor_orig = re.compile(r'^(?:Manoel|Manuel)\s+Joz[ée]\s+Frz\.?$', re.IGNORECASE)
    _manjor_false_doutor_expd = re.compile(r'^(?:Manoel|Manuel)\s+Jos[ée]\s+Fernandes$', re.IGNORECASE)
    for r in rows:
        if "doutor" not in r["Observações"].lower():
            continue
        if not (
            _manjor_false_doutor_orig.match(r["NomeOriginal"])
            or _manjor_false_doutor_expd.match(r["NomeAtualizado"])
        ):
            continue

        r["NomeOriginal"] = re.sub(r'^(?:Doutor\s+)?(?:Manoel|Manuel)\s+', '', r["NomeOriginal"], flags=re.IGNORECASE)
        r["NomeAtualizado"] = re.sub(r'^(?:Manoel|Manuel)\s+', '', r["NomeAtualizado"], flags=re.IGNORECASE)

        obs = re.sub(r'(?i)\b(?:o\s+)?doutor\b', '', r["Observações"])
        obs = re.sub(r'\s*;\s*;\s*', '; ', obs)
        obs = re.sub(r'^\s*;\s*|\s*;\s*$', '', obs).strip()
        r["Observações"] = "; ".join(filter(None, ["Major", obs]))

    # ------------------------------------------------------------------
    # Pós-processamento: criado/criada com "casado/a" → solteiro/a
    # O modelo confunde cr. (criado) com caz. (casado). Criados sem
    # estado civil explícito são sempre solteiros por omissão; "casado"
    # para criado/criada sem cônjuge no fogo é quase sempre leitura errada.
    # ------------------------------------------------------------------
    for r in rows:
        if (
            r["Parentesco"] in ("criado", "criada")
            and r["EstadoCivil"] in ("casado", "casada")
            and r["Fogo"] not in focos_com_ausente
        ):
            r["EstadoCivil"] = "solteira" if r["Sexo"] == "F" else "solteiro"

    # ------------------------------------------------------------------
    # Pós-processamento: "Cicrava" → "Escrava" (leitura errada de E→C, s→i)
    # O modelo confunde o E cursivo maiúsculo com C, produzindo "Cicrava"
    # em vez de "Escrava".
    # ------------------------------------------------------------------
    _cicrava_pat = re.compile(r'\bCicrava\b', re.IGNORECASE)
    for r in rows:
        if _cicrava_pat.search(r["Observações"]):
            r["Observações"] = _cicrava_pat.sub("Escrava", r["Observações"])

    # ------------------------------------------------------------------
    # Pós-processamento: título "Major" (oManjor/oMajor/Manjor/Major)
    # Remove o título do nome e regista em observações.
    # ------------------------------------------------------------------
    _major_prefix_orig = re.compile(r'^(?:o\s*)?manj(?:o|0)r\.?\s+', re.IGNORECASE)
    _major_prefix_expd = re.compile(r'^(?:o\s*)?major\s+', re.IGNORECASE)
    for r in rows:
        had_major_prefix = (
            bool(_major_prefix_orig.search(r["NomeOriginal"]))
            or bool(_major_prefix_expd.search(r["NomeAtualizado"]))
        )
        if not had_major_prefix:
            continue

        r["NomeOriginal"] = _major_prefix_orig.sub("", r["NomeOriginal"]).strip()
        r["NomeAtualizado"] = _major_prefix_expd.sub("", r["NomeAtualizado"]).strip()

        if "major" not in r["Observações"].lower():
            r["Observações"] = "; ".join(filter(None, ["Major", r["Observações"]]))

        # Evita conflito de títulos quando o erro inicial foi lido como "Doutor".
        if "doutor" in r["Observações"].lower():
            obs = re.sub(r'(?i)\b(?:o\s+)?doutor\b', '', r["Observações"])
            obs = re.sub(r'\s*;\s*;\s*', '; ', obs)
            r["Observações"] = re.sub(r'^\s*;\s*|\s*;\s*$', '', obs).strip()

    # ------------------------------------------------------------------
    # Pós-processamento: "aus.te" lido no nome ou não movido para observações
    # Transfere para observações e remove de nome. Se "menor" aparecer por
    # confusão com "m." ou erro de leitura do final, limpamos se for exclusivo.
    # ------------------------------------------------------------------
    _auste_pat = re.compile(r'\bau[sz]\.?t[ea]\b', re.IGNORECASE)
    for r in rows:
        has_auste = False
        if _auste_pat.search(r["NomeOriginal"]):
            r["NomeOriginal"] = _auste_pat.sub("", r["NomeOriginal"]).strip()
            r["NomeOriginal"] = re.sub(r'\s+', ' ', r["NomeOriginal"])
            r["NomeAtualizado"] = re.sub(r'(?i)\bausente\b', '', r["NomeAtualizado"]).strip()
            r["NomeAtualizado"] = _auste_pat.sub("", r["NomeAtualizado"]).strip()
            r["NomeAtualizado"] = re.sub(r'\s+', ' ', r["NomeAtualizado"])
            has_auste = True
            
        if _auste_pat.search(r["Observações"]):
            r["Observações"] = _auste_pat.sub("", r["Observações"]).strip()
            has_auste = True
            
        if has_auste:
            # Adicionar ausente
            if "ausente" not in r["Observações"].lower():
                r["Observações"] = "; ".join(filter(None, [r["Observações"], "ausente"]))
            # Remover "menor" se for erro provável por "aus.te"
            if "menor" in r["Observações"].lower():
                obs = re.sub(r'(?i)\bmenor\b', '', r["Observações"])
                obs = re.sub(r'\s*[,;]\s*[,;]\s*', ', ', obs)
                r["Observações"] = re.sub(r'^\s*[,;]\s*|\s*[,;]\s*$', '', obs).strip()

    # ------------------------------------------------------------------
    # Pós-processamento: irmão/irmã com "cunhado/a" nas observações
    # O modelo não detectou o marcador // e colocou a pessoa no fogo
    # anterior com parentesco "irmão". Corrige para "cabeça" e, se o
    # estado civil for "solteiro/a" (default de irmão), aplica a regra
    # de cabeça-sozinha → viúvo/viúva.
    # ------------------------------------------------------------------
    for r in rows:
        obs_lower = r["Observações"].lower()
        if r["Parentesco"] in ("irmão", "irmã") and "cunh" in obs_lower:
            r["Parentesco"] = "cabeça"
            if (
                r["EstadoCivil"] in ("solteiro", "solteira")
                and r["Fogo"] not in focos_com_mulher
                and r["Fogo"] not in focos_com_ausente
            ):
                r["EstadoCivil"] = "viúva" if r["Sexo"] == "F" else "viúvo"

    # ------------------------------------------------------------------
    # Pós-processamento: parentesco explícito escrito nas observações
    # Quando o modelo coloca graus de parentesco (sogra, sogro, genro, etc.)
    # em Observações para membros do mesmo fogo, mover para Parentesco.
    # ------------------------------------------------------------------
    _parentesco_expressoes = [
        "sogro", "sogra", "genro", "nora", "cunhado", "cunhada",
        "tio", "tia", "avô", "avó", "pai", "mãe", "sobrinho", "sobrinha", "afilhado", "afilhada",
    ]
    _parentesco_re = re.compile(r"\b(" + "|".join(_parentesco_expressoes) + r")\b", re.IGNORECASE)
    _sobr_abrev_re = re.compile(r"\bsobr\.?\s*([oaºª])\b", re.IGNORECASE)
    _afilh_abrev_re = re.compile(r"\bafilh\.?\s*([oaºª])\b", re.IGNORECASE)
    for r in rows:
        # Preservar cabeças de sub-fogo: nestes casos o grau pode ficar em observações.
        if r["Parentesco"] == "cabeça":
            continue

        # Normaliza formas abreviadas de sobrinho/a nas observações.
        r["Observações"] = _sobr_abrev_re.sub(
            lambda m: "sobrinha" if m.group(1).lower() in ("a", "ª") else "sobrinho",
            r["Observações"],
        )
        r["Observações"] = _afilh_abrev_re.sub(
            lambda m: "afilhada" if m.group(1).lower() in ("a", "ª") else "afilhado",
            r["Observações"],
        )

        m = _parentesco_re.search(r["Observações"])
        if not m:
            continue

        termo = m.group(1).lower()
        if r["Parentesco"] in ("", "desconhecido", "outro") or (
            termo in ("sobrinho", "sobrinha") and r["Parentesco"] in ("filho", "filha")
        ):
            r["Parentesco"] = termo
            # Remove o termo das observações e limpa separadores sobrantes.
            obs = _parentesco_re.sub("", r["Observações"])
            obs = re.sub(r"\s*;\s*;\s*", "; ", obs)
            obs = re.sub(r"^\s*;\s*|\s*;\s*$", "", obs).strip()
            r["Observações"] = obs

    # ------------------------------------------------------------------
    # Pós-processamento: evitar combinação impossível "mulher" + "mãe"
    # Se o parentesco já é "mulher" (esposa), qualquer marcador de mãe
    # nas observações é ruído de leitura e deve ser removido.
    # ------------------------------------------------------------------
    _mae_obs_re = re.compile(r"\b(mãe|mae|mãi|may|mai|maj)\b", re.IGNORECASE)
    for r in rows:
        if r["Parentesco"] != "mulher":
            continue
        if not _mae_obs_re.search(r["Observações"]):
            continue
        obs = _mae_obs_re.sub("", r["Observações"])
        obs = re.sub(r"\s*;\s*;\s*", "; ", obs)
        r["Observações"] = re.sub(r"^\s*;\s*|\s*;\s*$", "", obs).strip()

    # ------------------------------------------------------------------
    # Pós-processamento: Dem.te (demente) e sequência f.oDem.te
    # - Normaliza Dem.te/demte para "demente" em observações.
    # - Se há demente e o parentesco ficou outro/desconhecido, corrige para filho.
    # ------------------------------------------------------------------
    _demente_obs_re = re.compile(r"\b(?:dem\.?te|demente)\b", re.IGNORECASE)
    for r in rows:
        if _demente_obs_re.search(r["Observações"]):
            r["Observações"] = _demente_obs_re.sub("demente", r["Observações"])
            # Se houver ruído "menor" junto de Dem.te, prioriza demente.
            r["Observações"] = re.sub(r"(?i)\bmenor\b", "", r["Observações"])
            r["Observações"] = re.sub(r"\s*;\s*;\s*", "; ", r["Observações"])
            r["Observações"] = re.sub(r"^\s*;\s*|\s*;\s*$", "", r["Observações"]).strip()

            if r["Parentesco"] in ("", "outro", "desconhecido"):
                r["Parentesco"] = "filho"

    # ------------------------------------------------------------------
    # Pós-processamento: "Maximo" lido em vez de "Marinho"
    # O modelo lê frequentemente a grafia cursiva de "Marinho" como "Maximo".
    # Substituímos sempre no nome expandido.
    # ------------------------------------------------------------------
    _maximo_re = re.compile(r"\bMaximo\b", re.IGNORECASE)
    _ierey_re = re.compile(r"\bIerey\b", re.IGNORECASE)
    for r in rows:
        if bool(_maximo_re.search(r["NomeAtualizado"])):
            r["NomeAtualizado"] = _maximo_re.sub("Marinho", r["NomeAtualizado"])
        if bool(_ierey_re.search(r["NomeAtualizado"])):
            r["NomeAtualizado"] = _ierey_re.sub("Jesus", r["NomeAtualizado"])

    # ------------------------------------------------------------------
    # Pós-processamento: "de S. do Carmo" expandido incorretamente como
    # "de São do Carmo". Neste contexto, S. = Senhora.
    # ------------------------------------------------------------------
    _s_carmo_orig = re.compile(r'\bde\s+S\.\s+do\s+Carmo\b', re.IGNORECASE)
    _sao_carmo_expd = re.compile(r'\bde\s+S[aã]o\s+do\s+Carmo\b', re.IGNORECASE)
    for r in rows:
        if not _s_carmo_orig.search(r["NomeOriginal"]):
            continue
        if _sao_carmo_expd.search(r["NomeAtualizado"]):
            r["NomeAtualizado"] = _sao_carmo_expd.sub("de Senhora do Carmo", r["NomeAtualizado"])

    # ------------------------------------------------------------------
    # Pós-processamento: "Ingeitado/a" e "Enjeitado/a" não fazem parte do nome
    # Mantém o termo apenas em observações, com género coerente.
    # ------------------------------------------------------------------
    _enjeitado_nome_pat = re.compile(
        r'\b(?:ing[e]?itad[oa]|engeitad[oa]|enjeitad[oa])\b',
        re.IGNORECASE,
    )
    _enjeitado_obs_pat = re.compile(r'\benjeitad[oa]\b', re.IGNORECASE)
    for r in rows:
        found_in_nome = False

        if _enjeitado_nome_pat.search(r["NomeOriginal"]):
            r["NomeOriginal"] = _enjeitado_nome_pat.sub("", r["NomeOriginal"])
            r["NomeOriginal"] = re.sub(r'\s+', ' ', r["NomeOriginal"]).strip()
            found_in_nome = True

        if _enjeitado_nome_pat.search(r["NomeAtualizado"]):
            r["NomeAtualizado"] = _enjeitado_nome_pat.sub("", r["NomeAtualizado"])
            r["NomeAtualizado"] = re.sub(r'\s+', ' ', r["NomeAtualizado"]).strip()
            found_in_nome = True

        if not found_in_nome and not _enjeitado_obs_pat.search(r["Observações"]):
            continue

        termo = "enjeitada" if r["Sexo"] == "F" else "enjeitado"

        # Remove variações duplicadas nas observações e volta a inserir normalizado.
        obs_clean = re.sub(
            r'(?i)\b(?:ing[e]?itad[oa]|engeitad[oa]|enjeitad[oa])\b',
            '',
            r["Observações"],
        )
        obs_clean = re.sub(r'\s*[,;]\s*[,;]\s*', '; ', obs_clean)
        obs_clean = re.sub(r'^\s*[,;]\s*|\s*[,;]\s*$', '', obs_clean).strip()
        
        # Junta evitando vazios e substituindo eventuais vírgulas sozinhas esquecidas
        partes = [p.strip() for p in re.split(r'[,;]', obs_clean) if p.strip()]
        if termo not in partes:
            partes.append(termo)
        r["Observações"] = ", ".join(partes)

        if r["Parentesco"] in ("", "desconhecido"):
            r["Parentesco"] = "outro"

    # ------------------------------------------------------------------
    # Pós-processamento: resolver cadeia de sub-fogos
    # Quando o modelo devolve "//" correctamente para todos os sub-fogos,
    # todos já apontam para o mesmo # pai (last_hash_fogo nunca actualiza
    # em //). Este bloco corrige casos em que um sub-fogo aponta para outro
    # sub-fogo como pai (o seu pai imediato foi incorrectamente tratado como
    # // em vez de #, ou vice-versa): a cadeia é resolvida até ao # raiz.
    # ------------------------------------------------------------------
    _subfogo_re = re.compile(r"sub-fogo do fogo (\d+)")
    for r in rows:
        m = _subfogo_re.search(r["Observações"])
        if not m:
            continue
        parent = int(m.group(1))
        # Seguir a cadeia enquanto o pai for também um // sub-fogo
        visited: set[int] = {parent}
        while parent in fogo_parent:
            next_p = fogo_parent[parent]
            if next_p in visited:
                break  # protecção contra ciclos
            visited.add(next_p)
            parent = next_p
        original = int(m.group(1))
        if parent != original:
            r["Observações"] = _subfogo_re.sub(
                f"sub-fogo do fogo {parent}", r["Observações"]
            )

    # ------------------------------------------------------------------
    # Pós-processamento: sigla de apelido indeterminada no fim do nome
    # Quando aparece um apelido abreviado final sem correspondência no
    # CSV de siglas (ex.: "Carra.a"), ignora-se esse apelido no nome
    # expandido para evitar invenções como "Carrara".
    # ------------------------------------------------------------------
    known_siglas: set[str] = set()
    try:
        with open(CSV_MAPPING, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                for part in re.split(r"\s*/\s*", row.get("Sigla", "")):
                    token = re.sub(r"\s+", "", part).strip().lower()
                    if token:
                        known_siglas.add(token)
    except FileNotFoundError:
        pass

    _unknown_tail_sigla_pat = re.compile(r"([A-Za-zÀ-ÿ]{3,}\.[A-Za-zÀ-ÿªº])\s*$")
    for r in rows:
        m = _unknown_tail_sigla_pat.search(r["NomeOriginal"])
        if not m:
            continue
        sigla_final = re.sub(r"\s+", "", m.group(1)).lower()
        if sigla_final in known_siglas:
            continue
        nome_tokens = r["NomeAtualizado"].split()
        if len(nome_tokens) > 1:
            r["NomeAtualizado"] = " ".join(nome_tokens[:-1])

    return rows

# ====================================================================
# CHECKPOINT — guardar e retomar progresso
# ====================================================================
def save_checkpoint(pages_data: list[dict], current_lugar: str, processed: list[dict]) -> None:
    """Guarda o estado actual no ficheiro de checkpoint."""
    Path(CHECKPOINT_FILE).parent.mkdir(exist_ok=True)
    checkpoint = {
        "last_updated": datetime.now().isoformat(),
        "current_lugar": current_lugar,
        "processed": processed,      # lista de {ficheiro, pagina}
        "pages_data": pages_data,    # resultados acumulados
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, ensure_ascii=False, indent=2)


def load_checkpoint() -> dict | None:
    """Carrega o checkpoint se existir, caso contrário devolve None."""
    p = Path(CHECKPOINT_FILE)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def clear_checkpoint() -> None:
    """Remove o ficheiro de checkpoint após processamento completo."""
    p = Path(CHECKPOINT_FILE)
    if p.exists():
        p.unlink()


COLUMNS = [
    "Id", "Fogo", "Lugar", "NomeOriginal", "NomeAtualizado",
    "Sexo", "EstadoCivil", "Parentesco", "Confessou", "Observações",
]

def export_csv(rows: list[dict], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=";",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV guardado  → {output_path}")


def export_markdown(rows: list[dict], output_path: str) -> None:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep    = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"

    def cell(v):
        return str(v).replace("|", "\\|")

    body_lines = [
        "| " + " | ".join(cell(row.get(c, "")) for c in COLUMNS) + " |"
        for row in rows
    ]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = "\n".join([
        "# Rol de Confessados — Resultados",
        "",
        f"*Gerado em {ts} · {len(rows)} pessoas registadas*",
        "",
        header, sep,
        *body_lines,
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  Markdown guardado → {output_path}")

# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("  Processador de Manuscritos Genealógicos")
    print("=" * 60)

    # --- Validações iniciais ---
    if not API_KEY:
        print("\nERRO: GEMINI_API_KEY não definida.")
        print("  Cria o ficheiro .env com o conteúdo:")
        print("    GEMINI_API_KEY=a_tua_chave")
        print("  Obtém a chave em: https://aistudio.google.com/app/apikey")
        return

    input_path   = Path(INPUT_FOLDER)
    output_path  = Path(OUTPUT_FOLDER)
    mapping_path = Path(CSV_MAPPING)

    if not input_path.exists():
        print(f"\nERRO: Pasta de entrada não encontrada: {input_path}")
        return
    if not mapping_path.exists():
        print(f"\nERRO: Ficheiro de siglas não encontrado: {mapping_path}")
        return

    # --- Selecção de subpasta ---
    subpastas = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if subpastas:
        n_root = len(list(input_path.glob("*.pdf")))
        print(f"\nSubpastas encontradas em '{input_path}/':")
        print(f"  [0] (raiz)  —  {n_root} PDF(s) directamente em '{input_path}/'")
        for idx, sub in enumerate(subpastas, start=1):
            n_pdfs = len(list(sub.glob("*.pdf")))
            print(f"  [{idx}] {sub.name}/  —  {n_pdfs} PDF(s)")

        while True:
            escolha = input(f"\nEscolhe a pasta a processar [0-{len(subpastas)}]: ").strip()
            if escolha.isdigit() and 0 <= int(escolha) <= len(subpastas):
                break
            print(f"  Opção inválida. Introduz um número entre 0 e {len(subpastas)}.")

        escolha_idx = int(escolha)
        if escolha_idx > 0:
            input_path = subpastas[escolha_idx - 1]
            global CHECKPOINT_FILE
            CHECKPOINT_FILE = str(output_path / f"checkpoint_{input_path.name}.json")
            print(f"  A processar: {input_path}/")

    output_path.mkdir(exist_ok=True)

    # --- Verificar checkpoint existente ---
    checkpoint = load_checkpoint()
    all_pages_data: list[dict] = []
    processed_set: set[tuple]  = set()   # (ficheiro, pagina)
    current_lugar = ""

    if checkpoint:
        ts_cp = checkpoint.get("last_updated", "?")
        n_proc = len(checkpoint.get("processed", []))
        print(f"\nCheckpoint encontrado ({ts_cp})  —  {n_proc} página(s) já processada(s).")
        resposta = input("  Retomar de onde ficou? [S/n] ").strip().lower()
        if resposta != "n":
            all_pages_data = checkpoint.get("pages_data", [])
            current_lugar  = checkpoint.get("current_lugar", "")
            processed_set  = {(p["ficheiro"], p["pagina"])
                              for p in checkpoint.get("processed", [])}
            print(f"  A retomar. Lugar em vigor: '{current_lugar or 'não definido'}'")
        else:
            clear_checkpoint()
            print("  A recomeçar do início.")

    # --- Carregar siglas ---
    print(f"\nA carregar siglas de: {mapping_path}")
    siglas_text = load_siglas(str(mapping_path))
    prompt = build_prompt(siglas_text)
    print(f"  {len(siglas_text.splitlines())} entradas carregadas.")

    # --- Configurar Gemini ---
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"  Modelo: {GEMINI_MODEL}")

    # --- Listar PDFs por ordem natural ---
    pdf_files = sorted(input_path.glob("*.pdf"), key=natural_sort_key)
    if not pdf_files:
        print(f"\nERRO: Nenhum PDF encontrado em '{input_path}/'")
        return
    print(f"\nEncontrados {len(pdf_files)} ficheiros PDF em '{input_path}/'")

    # --- Processar cada PDF ---
    processed_list: list[dict] = list(checkpoint.get("processed", [])) if checkpoint else []
    total_pages   = len(processed_list)
    quota_atingida = False

    try:
        for pdf_file in pdf_files:
            pdf_nome = pdf_file.name
            try:
                images = pdf_to_images(str(pdf_file))
            except Exception as exc:
                print(f"\n  AVISO: Não foi possível abrir {pdf_nome} — {exc}")
                continue

            for i, image in enumerate(images, start=1):
                # Saltar páginas já processadas
                if (pdf_nome, i) in processed_set:
                    print(f"  [skip] {pdf_nome}  pág. {i}  (já processada)")
                    continue

                total_pages += 1
                idx = pdf_files.index(pdf_file) + 1
                print(f"\n  [{idx:>3}/{len(pdf_files)}] {pdf_nome}  pág. {i}/{len(images)}"
                      f"  (lugar: {current_lugar or '?'})  → Gemini...",
                      end=" ", flush=True)

                page_data = call_gemini(image, prompt, model, current_lugar)

                if page_data.get("lugar_atual"):
                    current_lugar = page_data["lugar_atual"].strip()

                # Actualiza também a partir de lugar_novo nos grupos (lugar a meio da página),
                # para que a próxima página receba o contexto correcto.
                for grupo in page_data.get("grupos", []):
                    lugar_novo = (grupo.get("lugar_novo") or "").strip()
                    if lugar_novo:
                        if re.search(r"doutra\s+par", lugar_novo, re.IGNORECASE):
                            current_lugar = current_lugar + " - Doutra Parte"
                        else:
                            current_lugar = lugar_novo

                # Indicador de lugar no fim da página (após todos os fogos)
                lugar_proximo = (page_data.get("lugar_proximo") or "").strip()
                if lugar_proximo:
                    current_lugar = lugar_proximo

                all_pages_data.append(page_data)
                processed_list.append({"ficheiro": pdf_nome, "pagina": i})
                processed_set.add((pdf_nome, i))

                n_p = sum(len(g.get("pessoas", [])) for g in page_data.get("grupos", []))
                n_g = len(page_data.get("grupos", []))
                print(f"{n_g} fogos, {n_p} pessoas  | lugar: {current_lugar or '?'}")

                # Guardar checkpoint após cada página
                save_checkpoint(all_pages_data, current_lugar, processed_list)

                # Pausa de cortesia entre pedidos
                if DELAY_ENTRE_PEDIDOS > 0:
                    time.sleep(DELAY_ENTRE_PEDIDOS)

    except QuotaDiariaEsgotada as exc:
        quota_atingida = True
        print(f"\n\n{'!'*60}")
        print(f"  QUOTA DIÁRIA ATINGIDA")
        print(f"  {exc}")
        print(f"  Progresso guardado em: {CHECKPOINT_FILE}")
        print(f"  Páginas processadas hoje: {len(processed_list)}")
        print(f"  Retoma amanhã correndo: python processar_manuscritos.py")
        print(f"{'!'*60}\n")

    if not all_pages_data:
        print("\nNenhum dado para exportar.")
        return

    # --- Construir tabela ---
    print(f"\nA construir tabela ({len(all_pages_data)} páginas processadas)...")
    rows = build_table(all_pages_data)
    print(f"  Total de pessoas registadas: {len(rows)}")

    # --- Exportar ---
    base_entrada = re.sub(r"[^0-9A-Za-z_-]+", "_", input_path.name).strip("_") or "manuscritos"
    sufixo  = "_parcial" if quota_atingida else ""
    csv_out = output_path / f"resultados_{base_entrada}{sufixo}.csv"
    md_out  = output_path / f"resultados_{base_entrada}{sufixo}.md"

    export_csv(rows, str(csv_out))
    export_markdown(rows, str(md_out))

    json_out = output_path / f"resultados_{base_entrada}_bruto{sufixo}.json"
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(all_pages_data, fh, ensure_ascii=False, indent=2)
    print(f"  JSON bruto guardado → {json_out}")

    # Limpar checkpoint só quando o processamento é completo
    if not quota_atingida:
        clear_checkpoint()
        print("\n" + "=" * 60)
        print("  Concluído! Checkpoint removido.")
        print("=" * 60)
    else:
        print(f"\n  Resultados parciais exportados. Checkpoint mantido para amanhã.")


if __name__ == "__main__":
    main()
