#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste rápido — Processador de Manuscritos Genealógicos
=======================================================
Processa apenas as primeiras N páginas e mostra o output detalhado
(JSON bruto do Gemini + tabela final) para validar o prompt e o
mapeamento antes de correr o processamento completo.

A pesquisa de PDFs é recursiva: procura em manuscritos/ e em todas
as subpastas (ex: manuscritos/1790/, manuscritos/1791/, etc.).

Uso:
    python testar_manuscritos.py                        # primeiros 5 PDFs (todas as subpastas)
    python testar_manuscritos.py 3                      # primeiros 3 PDFs
    python testar_manuscritos.py 1791-3                 # PDF cujo nome contenha "1791-3"
    python testar_manuscritos.py Rol - 1791-3.pdf       # fragmentos unidos automaticamente
    python testar_manuscritos.py 1971/Rol - 1791-3.pdf  # filtro por subpasta/nome
    python testar_manuscritos.py 1791-15 1791-16        # dois PDFs específicos
"""

import sys
import json
from pathlib import Path

# Importa tudo do script principal (reutiliza funções e configuração)
from processar_manuscritos import (
    QuotaDiariaEsgotada,
    natural_sort_key,
    load_siglas,
    pdf_to_images,
    build_prompt,
    call_gemini,
    build_table,
    export_csv,
    export_markdown,
    API_KEY,
    GEMINI_MODEL,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    CSV_MAPPING,
)

import google.generativeai as genai
from datetime import datetime

# ── Argumentos: número ou nomes parciais de PDFs ────────────────────
_args = sys.argv[1:]
if len(_args) == 1 and _args[0].isdigit():
    LIMITE_PDFS  = int(_args[0])
    FILTRO_NOMES = []
else:
    LIMITE_PDFS  = 5
    FILTRO_NOMES = _args   # lista de fragmentos de nomes (pode ser vazia)

# ────────────────────────────────────────────────────────────────────

def imprimir_pagina(pdf_nome: str, pagina: int, page_data: dict) -> None:
    """Imprime o JSON de uma página de forma legível para análise."""
    separator = "─" * 60

    lugar = page_data.get("lugar_atual")
    grupos = page_data.get("grupos", [])
    n_pessoas = sum(len(g.get("pessoas", [])) for g in grupos)

    print(f"\n{separator}")
    print(f"  {pdf_nome}  •  pág. {pagina}")
    if lugar:
        print(f"  📍 Lugar identificado: {lugar}")
    print(f"  {len(grupos)} fogos  |  {n_pessoas} pessoas")
    print(separator)

    for gi, grupo in enumerate(grupos, 1):
        simbolo = grupo.get("simbolo", "?")
        pessoas = grupo.get("pessoas", [])
        print(f"\n  Fogo {gi}  [{simbolo}]")
        for p in pessoas:
            conf  = "✓" if p.get("confessou") == "sim" else ("✗" if p.get("confessou") == "não" else "?")
            sexo  = p.get("sexo", "?")
            role  = p.get("parentesco", "?")
            ec    = p.get("estado_civil", "")
            obs   = p.get("observacoes", "")
            orig  = p.get("nome_original", "")
            expd  = p.get("nome_expandido", "")

            role_str = f"{role}"
            if ec and ec != "desconhecido":
                role_str += f" / {ec}"
            obs_str = f"  ← {obs}" if obs else ""

            print(f"    [{conf}] {sexo}  {orig:<35}  →  {expd:<35}  ({role_str}){obs_str}")

    print()


def main():
    print("=" * 60)
    print("  TESTE — Manuscritos Genealógicos")
    if FILTRO_NOMES:
        print(f"  PDFs seleccionados: {', '.join(FILTRO_NOMES)}")
    else:
        print(f"  A processar os primeiros {LIMITE_PDFS} PDFs")
    print("=" * 60)

    if not API_KEY:
        print("\nERRO: GEMINI_API_KEY não definida. Cria o ficheiro .env com a chave.")
        return

    input_path  = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER) / "teste"
    output_path.mkdir(parents=True, exist_ok=True)

    # Carregar siglas e construir prompt
    print(f"\nA carregar siglas de: {CSV_MAPPING}")
    siglas_text = load_siglas(CSV_MAPPING)
    prompt = build_prompt(siglas_text)
    print(f"  {len(siglas_text.splitlines())} entradas carregadas.")

    # Configurar Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"  Modelo: {GEMINI_MODEL}\n")

    # Seleccionar PDFs
    todos_pdfs = sorted(input_path.glob("**/*.pdf"), key=natural_sort_key)
    if FILTRO_NOMES:
        # Tenta primeiro com todos os fragmentos unidos (lida com espaços no nome)
        joined = " ".join(FILTRO_NOMES)
        # Compara contra o nome do ficheiro E contra o caminho relativo (suporta "1971/Rol...")
        pdf_files = [
            f for f in todos_pdfs
            if joined.lower() in f.name.lower()
            or joined.lower() in str(f.relative_to(input_path)).lower()
        ]
        # Se não houver correspondência, trata cada fragmento como identificador separado
        if not pdf_files:
            pdf_files = [
                f for f in todos_pdfs
                if any(
                    frag.lower() in f.name.lower()
                    or frag.lower() in str(f.relative_to(input_path)).lower()
                    for frag in FILTRO_NOMES
                )
            ]
        if not pdf_files:
            print(f"ERRO: Nenhum PDF encontrado com os filtros: {', '.join(FILTRO_NOMES)}")
            print("PDFs disponíveis:")
            for f in todos_pdfs:
                print(f"  • {f.relative_to(input_path)}")
            return
    else:
        pdf_files = todos_pdfs[:LIMITE_PDFS]
        if not pdf_files:
            print(f"ERRO: Nenhum PDF encontrado em '{input_path}/'")
            return
    print(f"PDFs seleccionados para teste ({len(pdf_files)}):")
    for f in pdf_files:
        print(f"  • {f.relative_to(input_path)}")

    # Processar
    all_pages_data = []
    raw_responses  = []   # guarda o JSON bruto para análise
    current_lugar  = ""   # memória do lugar entre páginas

    try:
      for pdf_file in pdf_files:
        print(f"\n{'═'*60}")
        print(f"  {pdf_file.name}")
        print(f"{'═'*60}")
        try:
            images = pdf_to_images(str(pdf_file))
        except Exception as exc:
            print(f"  AVISO: Não foi possível abrir o PDF — {exc}")
            continue

        for i, image in enumerate(images, start=1):
            print(f"\n  A enviar página {i}/{len(images)} ao Gemini... (lugar: {current_lugar or 'não definido'})", flush=True)
            page_data = call_gemini(image, prompt, model, current_lugar)
            # Actualiza a memória do lugar para a próxima página
            if page_data.get("lugar_atual"):
                current_lugar = page_data["lugar_atual"].strip()
            all_pages_data.append(page_data)
            raw_responses.append({
                "ficheiro": pdf_file.name,
                "pagina": i,
                "lugar_em_vigor": current_lugar,
                "resposta": page_data,
            })
            imprimir_pagina(pdf_file.name, i, page_data)

    except QuotaDiariaEsgotada as exc:
        print(f"\n{'!'*60}")
        print(f"  QUOTA DIÁRIA ATINGIDA — resultados parciais serão exportados")
        print(f"  {exc}")
        print(f"{'!'*60}\n")

    # Construir tabela final
    print("=" * 60)
    print("  TABELA FINAL  (após correcções automáticas)")
    print("=" * 60)
    rows = build_table(all_pages_data)

    # Mostrar tabela no formato [✓] com dados já corrigidos
    fogo_atual = None
    for row in rows:
        if row["Fogo"] != fogo_atual:
            fogo_atual = row["Fogo"]
            print(f"\n  Fogo {fogo_atual}")
        conf     = "✓" if row["Confessou"] == "sim" else ("✗" if row["Confessou"] == "não" else "?")
        sexo     = row["Sexo"]
        orig     = row["NomeOriginal"]
        expd     = row["NomeAtualizado"]
        role_str = row["Parentesco"]
        ec       = row["EstadoCivil"]
        obs      = row["Observações"]
        if ec and ec != "desconhecido":
            role_str += f" / {ec}"
        obs_str = f"  ← {obs}" if obs else ""
        print(f"    [{conf}] {sexo}  {orig:<35}  →  {expd:<35}  ({role_str}){obs_str}")

    print(f"\n  Total: {len(rows)} pessoas  |  {max((r['Fogo'] for r in rows), default=0)} fogos")

    # Exportar resultados de teste
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    export_csv(rows, str(output_path / f"teste_{ts}.csv"))
    export_markdown(rows, str(output_path / f"teste_{ts}.md"))

    # Guardar JSON bruto completo para inspeção
    json_out = output_path / f"teste_bruto_{ts}.json"
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(raw_responses, fh, ensure_ascii=False, indent=2)
    print(f"  JSON bruto guardado → {json_out}")

    print("\n" + "=" * 60)
    print("  Revê os resultados em saida/teste/")
    print("  Se o output estiver correcto, corre: python processar_manuscritos.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
