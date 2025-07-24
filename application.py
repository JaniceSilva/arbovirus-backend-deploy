#!/usr/bin/env python3
"""
Arquivo principal para AWS Elastic Beanstalk
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import app

application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
