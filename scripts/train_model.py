from echelon.utils.logging import get_logger
from echelon.data.manager import ThreatDataManager
from echelon.ml.model import ThreatMLModel

logger = get_logger(__name__)

def main():
    data_manager = ThreatDataManager()
    
    logger.info("Initializing ML model...")
    ml_model = ThreatMLModel(data_manager)
    
    logger.info("Training ML model...")
    success = ml_model.train_model()
    
    if success:
        logger.info(f"Model training successful. Accuracy: {ml_model.accuracy:.2f}")
        
        # Generate sample predictions
        predictions = ml_model.predict_threats(num_predictions=3)
        logger.info(f"Generated {len(predictions)} sample predictions:")
        
        for i, prediction in enumerate(predictions):
            logger.info(f"Prediction {i+1}: {prediction['apt_group']} - {prediction['attack_type']} - {prediction['confidence']}%")
    else:
        logger.warning("Model training failed.")

if __name__ == "__main__":
    main()
