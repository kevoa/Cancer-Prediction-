from CancerPrediction.config.configuration import ConfigurationManager
from CancerPrediction.components.model_trainer import ModelTrainer
from CancerPrediction import logger
import pandas as pd
from pathlib import Path

STAGE_NAME = "Model Training"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            
            config_manager = ConfigurationManager()
            
            model_trainer_config = config_manager.get_model_trainer_config()
            model_params = {
                'RandomForest': config_manager.get_params('RandomForest'),
                'GradientBoosting': config_manager.get_params('GradientBoosting'),
                'LightGBM': config_manager.get_params('LightGBM'),
                'XGBoost': config_manager.get_params('XGBoost'),
                'CTGAN': config_manager.get_params('CTGAN')
            }
            
            model_trainer = ModelTrainer(config=model_trainer_config, params=model_params)
            
            model_trainer.train()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
            
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
