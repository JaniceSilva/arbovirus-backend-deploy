#!/bin/bash

# Criar diretório de banco de dados se não existir
mkdir -p /var/app/current/src/database

# Definir permissões
chmod 755 /var/app/current/src/database

# Executar migração do banco de dados
cd /var/app/current
python src/database/migrations.py create || true

