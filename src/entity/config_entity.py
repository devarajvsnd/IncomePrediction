from collections import namedtuple


DataIngestionConfig=namedtuple("DataIngestionConfig", ["dataset_download_url","tgz_download_dir","raw_data_dir"])


DataTransformationConfig = namedtuple("DataTransformationConfig", ["schema_file_path", "transformed_data_dir"])


ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path","base_accuracy","model_config_file_path"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])
