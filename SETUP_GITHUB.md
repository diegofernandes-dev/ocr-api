# Configuração do Repositório GitHub

## Status Atual

✅ **Repositório Git local**: Configurado e com commits
✅ **Repositório GitHub**: Criado e sincronizado (https://github.com/diegofernandes-dev/ocr-api)
✅ **Imagem Docker**: Construída e publicada no Docker Hub (`diegoistta/ocr-api:latest`)

## Repositórios

- **GitHub**: https://github.com/diegofernandes-dev/ocr-api
- **Docker Hub**: https://hub.docker.com/r/diegoistta/ocr-api

## Comandos úteis

```bash
# Verificar status do Git
git status

# Verificar remote configurado
git remote -v

# Fazer push após criar o repositório no GitHub
git push -u origin main

# Executar a imagem Docker localmente
docker run -p 8080:8080 diegoistta/ocr-api:latest

# Fazer pull da imagem do Docker Hub
docker pull diegoistta/ocr-api:latest
```

## Estrutura do Projeto

- `ocr_service.py`: Serviço principal da API OCR
- `requirements.txt`: Dependências Python
- `Dockerfile`: Configuração para containerização
- `README.md`: Documentação do projeto
- `.gitignore`: Arquivos a serem ignorados pelo Git 