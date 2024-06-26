# src/CancerPrediction/pipeline/stages_02_data_validation.py
from CancerPrediction.config.configuration import ConfigurationManager
from CancerPrediction.components.data_validation import DataValidation
from CancerPrediction import logger
import pandas as pd

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        # Inicializar el ConfigurationManager
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        
        # Cargar datos desde el archivo de ingestión
        df = pd.read_excel(data_validation_config.unzip_data_dir)
        logger.info("DataFrame loaded successfully")
        
        # Crear instancia de DataValidation
        data_validation = DataValidation(df=df, config=data_validation_config)
        
        # Ejecutar la validación completa
        df_validated = data_validation.validate()
        logger.info("Data validation completed successfully")

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
