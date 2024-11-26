import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from streamlit_app import predict_crop, predict_nutrients, crop_dict

# Mock the models
@pytest.fixture
def mock_models():
    # Mocking the scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])

    # Mocking the crop nutrient model
    mock_crop_nutrient_model = MagicMock()
    mock_crop_nutrient_model.predict.return_value = np.array([2.0, 5.0, 3.0, 30.0, 70.0, 6.5, 150.0])  # Mocked nutrient values

    # Mocking the crop encoder
    mock_crop_encoder = MagicMock()
    mock_crop_encoder.transform.return_value = np.array([1])

    # Mocking the best_rfc (Random Forest Classifier)
    mock_best_rfc = MagicMock()
    mock_best_rfc.predict.return_value = np.array([0])  # Mocked crop prediction index (0 corresponds to 'Rice')

    # Return all the mocked objects as a dictionary
    return {
        'scaler': mock_scaler,
        'crop_nutrient_model': mock_crop_nutrient_model,
        'crop_encoder': mock_crop_encoder,
        'best_rfc': mock_best_rfc
    }

# Test for the crop prediction function
def test_predict_crop(mock_models):
    # Extracting the mocked models from the fixture
    scaler = mock_models['scaler']
    best_rfc = mock_models['best_rfc']
    
    # Predict crop using mocked models
    predicted_crop = predict_crop(10.0, 5.0, 12.0, 30.0, 80.0, 6.5, 120.0)
    
    # Assert that the predicted crop matches the mocked result ('Rice')
    assert predicted_crop == "Rice"
    
    # Check if the scaler's transform method was called
    scaler.transform.assert_called_once()

# Test for the nutrient recommendation function
def test_predict_nutrients(mock_models):
    # Extracting the mocked models from the fixture
    crop_encoder = mock_models['crop_encoder']
    crop_nutrient_model = mock_models['crop_nutrient_model']
    
    # Predict nutrients for 'Rice'
    nutrients = predict_nutrients("Rice")
    
    # Assert that the nutrient prediction matches the mocked values
    assert np.array_equal(nutrients, np.array([2.0, 5.0, 3.0, 30.0, 70.0, 6.5, 150.0]))

    # Check if the crop_encoder's transform method was called
    crop_encoder.transform.assert_called_once()

# Test for invalid crop name
def test_predict_nutrients_invalid_crop(mock_models):
    # Test the invalid crop name case
    nutrients_invalid = predict_nutrients("InvalidCrop")
    assert nutrients_invalid == "Invalid crop name. Please enter a valid crop."
