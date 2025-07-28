# Configuração do Repositório GitHub

## Passos para criar o repositório no GitHub:

1. **Acesse o GitHub**: Vá para https://github.com/diegofernandes-dev

2. **Crie um novo repositório**:
   - Clique no botão "New" ou "Novo repositório"
   - Nome do repositório: `ocr_api`
   - Descrição: `OCR API service with Docker support`
   - Deixe como público ou privado conforme sua preferência
   - **NÃO** inicialize com README, .gitignore ou licença (já temos esses arquivos)

3. **Após criar o repositório**, execute os seguintes comandos no terminal:

```bash
# Fazer push para o GitHub
git push -u origin main
```

## Status Atual

✅ **Repositório Git local**: Configurado e com commits
✅ **Imagem Docker**: Construída e publicada no Docker Hub (`diegoistta/ocr_api:latest`)
⏳ **Repositório GitHub**: Aguardando criação manual

## Comandos úteis

```bash
# Verificar status do Git
git status

# Verificar remote configurado
git remote -v

# Fazer push após criar o repositório no GitHub
git push -u origin main

# Executar a imagem Docker localmente
docker run -p 8080:8080 diegoistta/ocr_api:latest

# Fazer pull da imagem do Docker Hub
docker pull diegoistta/ocr_api:latest
```

## Estrutura do Projeto

- `ocr_service.py`: Serviço principal da API OCR
- `requirements.txt`: Dependências Python
- `Dockerfile`: Configuração para containerização
- `README.md`: Documentação do projeto
- `.gitignore`: Arquivos a serem ignorados pelo Git 