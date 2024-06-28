from CancerPrediction.config.configuration import ConfigurationManager
from CancerPrediction.components.data_transformation import DataTransformation
import pandas as pd
from CancerPrediction import logger
from pathlib import Path
import logging


STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try: 
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
                if status == "True":
                    # Inicializar el ConfigurationManager
                    config = ConfigurationManager()
                    
                    # Obtener la configuración de transformación de datos
                    data_transformation_config = config.get_data_transformation_config()
                    
                    # Cargar datos desde el archivo validado
                    df = pd.read_excel(data_transformation_config.validated_data_file)
                    print("DataFrame loaded successfully:")
                    print(df.head())
                    
                    # Crear instancia de DataTransformation
                    data_transformation = DataTransformation(df, data_transformation_config)
                    
                    # Ejecutar la transformación completa
                    data_transformation.transform()
                    print("Transformation stage completed successfully")
                
                else:
                    raise Exception("Your data schema is not valid")
        except Exception as e:
            logger.error(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
