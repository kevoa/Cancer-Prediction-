from CancerPrediction.constants import *
from CancerPrediction.utils.common import read_yaml, create_directories
from CancerPrediction.entity.config_entity import (DataIngestionConfig,
                                                   DataValidationConfig, DataTransformationConfig, ModelTrainerConfig)
class ConfigurationManager:
    def __init__(
        self, 
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([Path(self.config['artifacts_root'])])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir'])
        )
        
        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        schema = self.schema['COLUMNS']
        
        create_directories([Path(config['root_dir'])])
        
        return DataValidationConfig(
            root_dir=Path(config['root_dir']),
            STATUS_FILE=config['status_file'],
            unzip_data_dir=Path(config['data_file']),
            all_schema=schema,
            sequences_to_remove=config['sequences_to_remove'],
            target_column=config['target_column'],
            columns_to_remove=config['columns_to_remove']
        )
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        schema = self.schema['COLUMNS']
        
        create_directories([Path(config['root_dir'])])
        
        return DataTransformationConfig(
            root_dir=Path(config['root_dir']),
            validated_data_file=Path(config['validated_data_file']),
            transformed_train_data_path=Path(config['transformed_train_data_path']),
            transformed_test_data_path=Path(config['transformed_test_data_path']),
            target_column=config['target_column'],
            ordinal_features=config['ordinal_features'],
            nominal_features=config['nominal_features']
        )
        

    def get_model_trainer_config(self) -> ModelTrainerConfig:
            config = self.config['model_trainer']
            
            create_directories([Path(config['root_dir'])])
            
            return ModelTrainerConfig(
                root_dir=Path(config['root_dir']),
                train_data_path=Path(config['train_data_path']),
                test_data_path=Path(config['test_data_path']),
                model_name=config['model_name'],
                important_features=config['important_features'],
                target_column=config['target_column']
            )