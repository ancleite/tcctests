# Use uma imagem base do Python
FROM python:3.9-slim

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie os arquivos do projeto para o diretório de trabalho
COPY . /app

# Instale as dependências
RUN pip install -r requirements.txt

# Exponha a porta onde a aplicação irá rodar
CMD ["pnpm", "run", "preview", "--host", "0.0.0.0", "--port", "8080"]
EXPOSE 8080

# Defina a variável de ambiente PORT
ENV PORT 8080

# Comando para rodar a aplicação
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
