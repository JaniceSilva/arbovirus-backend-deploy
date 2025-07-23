#!/usr/bin/env python3
"""
Arquivo principal para AWS Elastic Beanstalk
"""
import os
import sys

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import app

# Elastic Beanstalk espera uma variável chamada 'application'
application = app

if __name__ == '__main__':
    # Para desenvolvimento local
    application.run(host='0.0.0.0', port=5000, debug=False)

