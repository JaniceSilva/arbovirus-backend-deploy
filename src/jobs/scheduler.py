"""
Script de agendamento de jobs para AWS Lambda/EventBridge
"""
import os
import sys
import json
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.jobs.data_collector import DataCollector
from src.jobs.prediction_processor import PredictionProcessor

def lambda_handler_daily_collection(event, context):
    """
    Handler para AWS Lambda - Coleta diária de dados
    
    Args:
        event: Evento do EventBridge
        context: Contexto do Lambda
        
    Returns:
        Resposta HTTP
    """
    try:
        print(f"🚀 Iniciando job diário via Lambda - {datetime.now()}")
        
        collector = DataCollector()
        collector.run_daily_collection()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Coleta diária executada com sucesso',
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        print(f"❌ Erro no job diário: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def lambda_handler_monthly_prediction(event, context):
    """
    Handler para AWS Lambda - Processamento mensal de predições
    
    Args:
        event: Evento do EventBridge
        context: Contexto do Lambda
        
    Returns:
        Resposta HTTP
    """
    try:
        print(f"🚀 Iniciando job mensal via Lambda - {datetime.now()}")
        
        processor = PredictionProcessor()
        processor.run_monthly_prediction()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processamento mensal executado com sucesso',
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        print(f"❌ Erro no job mensal: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def run_local_daily():
    """Executar job diário localmente"""
    collector = DataCollector()
    collector.run_daily_collection()

def run_local_monthly():
    """Executar job mensal localmente"""
    processor = PredictionProcessor()
    processor.run_monthly_prediction()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python scheduler.py [daily|monthly]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'daily':
        run_local_daily()
    elif command == 'monthly':
        run_local_monthly()
    else:
        print("Comando inválido. Use: daily ou monthly")
        sys.exit(1)

