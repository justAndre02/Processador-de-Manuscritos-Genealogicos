# RESULTADO DO TESTE build_table COM DADOS MÍNIMOS

**Data:** 2024
**Script:** test_build_table.py
**Função Testada:** build_table() do módulo processar_manuscritos.py

---

## 📋 DADOS DE ENTRADA

Uma pessoa com os seguintes dados:
- **nome_original:** `"D. Antonio Jozé de Freitas"`
- **nome_expandido:** `"D. António José de Freitas"`
- **sexo:** `"M"` (Masculino)
- **estado_civil:** `"solteiro"`
- **parentesco:** `"filho"`
- **confessou:** `"Sopti"` (variante paleográfica de confessão)
- **observacoes:** `""` (vazio)

---

## ✅ VERIFICAÇÕES REALIZADAS

### 1️⃣ SE 'D.' FOI REMOVIDO DO NOME

**Status:** ✅ **PASSOU**

- **NomeOriginal Inicial:** `"D. Antonio Jozé de Freitas"`
- **NomeOriginal Final:** `"Antonio Jozé de Freitas"`
- **Resultado:** `✓ 'D.' foi REMOVIDO com sucesso`

- **NomeAtualizado Inicial:** `"D. António José de Freitas"`
- **NomeAtualizado Final:** `"António José de Freitas"`
- **Resultado:** `✓ 'D.' foi REMOVIDO com sucesso`

**Motivo:** O sistema detectou que `D.` + `"Sopti"` é uma leitura errada de `"o P.e"` (padre), acionando o pós-processamento nas linhas 893-907 que remove automaticamente o prefixo.

---

### 2️⃣ SE 'PADRE' ESTÁ EM OBSERVAÇÕES

**Status:** ✅ **PASSOU**

- **Observações Inicial:** `""` (vazio)
- **Observações Intermediária (após Sopti):** `"Sopti"`
- **Observações Final:** `"Padre; Sopti"`
- **Resultado:** `✓ 'Padre' FOI ADICIONADO com sucesso`

**Motivo:** Quando o sistema remove `D.`, adiciona automaticamente `"Padre"` ao início das observações (linha 906), preservando também a evidência `"Sopti"`.

---

### 3️⃣ SE CONFESSOU='SIM'

**Status:** ✅ **PASSOU**

- **Confessou Inicial:** `"Sopti"` (sigla/variante)
- **Confessou Final:** `"sim"` (normalizado)
- **Resultado:** `✓ Confessou FOI NORMALIZADO para 'sim' com sucesso`

**Motivo:** O pós-processamento de confissão (linhas 806-817) reconhece que `"Sopti"` é uma variante válida de confissão e normaliza automaticamente para `"sim"`, movendo o valor original para observações como prova.

---

## 🎯 RESULTADO FINAL COMPLETO

| Campo | Valor Inicial | Valor Final |
|-------|---------------|-------------|
| **NomeOriginal** | `D. Antonio Jozé de Freitas` | `Antonio Jozé de Freitas` ✓ |
| **NomeAtualizado** | `D. António José de Freitas` | `António José de Freitas` ✓ |
| **Sexo** | `M` | `M` |
| **EstadoCivil** | `solteiro` | `solteiro` |
| **Parentesco** | `filho` | `filho` |
| **Confessou** | `Sopti` | `sim` ✓ |
| **Observações** | `` (vazio) | `Padre; Sopti` ✓ |

---

## 📊 SUMÁRIO DE TESTES

```
✅ TESTE 1: NomeAtualizado SEM 'D.'      → PASSOU
✅ TESTE 2: Observações COM 'Padre'      → PASSOU
✅ TESTE 3: Confessou='sim'              → PASSOU

✅ TODOS OS TESTES PASSARAM COM SUCESSO!
```

---

## 🔍 ANÁLISE TÉCNICA

### Fluxo de Processamento

1. **Fase 1: Construção Inicial (linhas 782-793)**
   - Cria a linha com os dados exatamente como fornecidos
   - `Confessou = "Sopti"`, `Observações = ""`

2. **Fase 2: Pós-processamento Confessou (linhas 806-817)**
   - Detecta `"Sopti"` como variante válida de confissão
   - Normaliza para `"sim"`
   - Move original para observações: `Observações = "Sopti"`

3. **Fase 3: Pós-processamento "oP.e" como "D." (linhas 893-907)**
   - **Condição 1:** Sexo = "M" ✓
   - **Condição 2:** Nome começa com "D. " ✓
   - **Condição 3:** Campo Confessou/Observações contém "Sopti" ✓
   - **Ação:** Remove "D." dos nomes e adiciona "Padre" às observações

### Implementação em Regex

```python
# Detecção de prefixo D.
_d_prefix = re.compile(r'^D\.\s+', re.IGNORECASE)

# Detecção de Sopti/Spti
re.search(r"(?i)\bsopti\b|\bspti\b", marker_text)

# Remoção e substituição
r["NomeOriginal"] = _d_prefix.sub("", r["NomeOriginal"]).strip()
r["NomeAtualizado"] = _d_prefix.sub("", r["NomeAtualizado"]).strip()

# Adição de Padre
r["Observações"] = "; ".join(filter(None, ["Padre", r["Observações"]]))
```

---

## 💡 INTERPRETAÇÃO

O sistema funcionou exatamente como esperado:

- **"D. Sopti" → Padre identificado:** O modelo Gemini frequentemente lê "o P.e" (padre) como "D." (Dom/Dona), combinado com "Sopti" (variante paleográfica de confissão). O pós-processamento detecta este padrão comum e corrige automaticamente.

- **Normalização de Confessou:** Qualquer variante de confissão ("Sopti", "Spti", "Conf", etc.) é normalizada para "sim" para facilitar análise estatística, mantendo a evidência original em observações.

- **Limpeza de Nomes:** Títulos, prefixos de tratamento e leituras erradas de símbolos paleográficos são removidos dos nomes para manter integridade de dados genealógicos.

---

## ✨ CONCLUSÃO

✅ **A função `build_table` está funcionando CORRETAMENTE com todos os critérios de teste sendo satisfeitos.**
