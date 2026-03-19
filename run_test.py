#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simples para testar build_table com dados mínimos
"""
import sys
sys.path.insert(0, r'c:\Users\Andre\Documents\GitHub\Processador-de-Manuscritos-Genealogicos')

import json
from processar_manuscritos import build_table

# Dados de teste
pages_data = [
    {
        "lugar_atual": "Teste",
        "grupos": [
            {
                "simbolo": "#",
                "lugar_novo": None,
                "pessoas": [
                    {
                        "nome_original": "D. Antonio Jozé de Freitas",
                        "nome_expandido": "D. António José de Freitas",
                        "sexo": "M",
                        "estado_civil": "solteiro",
                        "parentesco": "filho",
                        "confessou": "Sopti",
                        "observacoes": ""
                    }
                ]
            }
        ]
    }
]

# Executar build_table
resultado = build_table(pages_data)

# Exibir resultado
print("=" * 80)
print("RESULTADO DA FUNÇÃO build_table")
print("=" * 80)
print(json.dumps(resultado, indent=2, ensure_ascii=False))
print("=" * 80)

if resultado:
    pessoa = resultado[0]
    print("\nVERIFICAÇÃO:")
    print("-" * 80)
    print(f"Nome Final: {pessoa.get('NomeAtualizado', 'N/A')}")
    print(f"  ✓ Tem 'D.'? {('D.' in pessoa.get('NomeAtualizado', ''))}")
    print(f"  ✗ NÃO tem 'D.'? {('D.' not in pessoa.get('NomeAtualizado', ''))}")
    
    print(f"\nObservações: {pessoa.get('Observações', 'N/A')}")
    print(f"  ✓ Tem 'Padre'? {('Padre' in pessoa.get('Observações', ''))}")
    
    print(f"\nConfessou: {pessoa.get('Confessou', 'N/A')}")
    print(f"  ✓ É 'sim'? {pessoa.get('Confessou') == 'sim'}")
    
    print("\n" + "=" * 80)
    print("RESULTADO FINAL:")
    print("=" * 80)
    
    # Verificações
    sem_d_ponto = 'D.' not in pessoa.get('NomeAtualizado', '')
    tem_padre = 'Padre' in pessoa.get('Observações', '')
    confessou_sim = pessoa.get('Confessou') == 'sim'
    
    print(f"✓ NomeAtualizado SEM 'D.': {sem_d_ponto}")
    print(f"✓ Observações COM 'Padre': {tem_padre}")
    print(f"✓ Confessou='sim': {confessou_sim}")
    
    if sem_d_ponto and tem_padre and confessou_sim:
        print("\n✓ TODOS OS TESTES PASSARAM!")
        sys.exit(0)
    else:
        print("\n✗ ALGUNS TESTES FALHARAM!")
        sys.exit(1)
else:
    print("✗ ERRO: build_table retornou lista vazia!")
    sys.exit(1)
