from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", ["data_file_path", "is_ingested", "message"])


DataTransformationArtifact = namedtuple("DataTransformationArtifact", 
                                        ["is_transformed", "message", "transformed_data_file_path"])


ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", "message", "trained_model_file_path",
                                                            "train_accuracy", "test_accuracy", "model_accuracy"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pusher", "export_model_file_path"])
