import json
import os
import glob
import re

def get_ordered_files(base_dir):
    dirs = [
        "utils",
        "data",
        "preprocessing",
        "features",
        "models",
        "training",
        "evaluation",
        "inference",
        "persistence"
    ]
    files = []
    
    config = os.path.join(base_dir, "config.py")
    if os.path.exists(config):
        files.append(config)
        
    for d in dirs:
        pattern = os.path.join(base_dir, d, "*.py")
        for f in sorted(glob.glob(pattern)):
            if not f.endswith("__init__.py"):
                files.append(f)
                
    pipeline = os.path.join(base_dir, "pipeline.py")
    if os.path.exists(pipeline):
        files.append(pipeline)
        
    return files

def clean_code(content):
    lines = content.split('\n')
    new_lines = []
    in_import = False
    
    for line in lines:
        if in_import:
            if ')' in line:
                in_import = False
            continue
            
        # Match "from ml..." or "from ml.triclass..."
        if re.match(r'^\s*from\s+ml[\.\w]*\s+import\b', line):
            if '(' in line and ')' not in line:
                in_import = True
            continue
            
        if re.match(r'^\s*import\s+ml[\.\w]*\b', line):
            continue
            
        new_lines.append(line)
        
    return '\n'.join(new_lines)

def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in text.split('\n')]
    }

def make_code_cell(code):
    source = [line + '\n' for line in code.split('\n')]
    if source:
        source[-1] = source[-1].rstrip('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def main():
    binary_files = get_ordered_files("ml")
    triclass_files = get_ordered_files("ml/triclass")
    
    cells = []
    
    cells.append(make_markdown_cell("# Projeto ML - Detecção de DDoS em SDN\n\nEste notebook contém todo o código do diretório `ml/`, unificado seguindo o guia de boas práticas."))
    
    cells.append(make_markdown_cell("## Parte 1: Pipeline de Classificação Binária"))
    for f in binary_files:
        if "triclass" in f: continue
        cells.append(make_markdown_cell(f"### Arquivo: `{f}`"))
        with open(f, "r", encoding="utf-8") as file:
            code = clean_code(file.read())
            cells.append(make_code_cell(code))
            
    # Add execution cell for binary
    cells.append(make_markdown_cell("### Execução do Pipeline Binário\nDescomente a célula abaixo para executar:"))
    cells.append(make_code_cell("# run_pipeline(run_tuning=False, run_eda=True, run_id='notebook_binary')"))
            
    cells.append(make_markdown_cell("## Parte 2: Pipeline de Classificação Triclasse"))
    for f in triclass_files:
        cells.append(make_markdown_cell(f"### Arquivo: `{f}`"))
        with open(f, "r", encoding="utf-8") as file:
            code = clean_code(file.read())
            cells.append(make_code_cell(code))
            
    # Add execution cell for triclass
    cells.append(make_markdown_cell("### Execução do Pipeline Triclasse\nDescomente a célula abaixo para executar:"))
    cells.append(make_code_cell("# run_triclass_pipeline(run_tuning=False, run_eda=True, save_plots=True, run_id='notebook_triclass')"))
            
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open("ml_notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
        
    print("Notebook 'ml_notebook.ipynb' criado com sucesso.")

if __name__ == "__main__":
    main()
