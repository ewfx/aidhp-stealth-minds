import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app import load_data, check_bias, run_benchmarking, generate_recommendation

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'customer_id': ['1', '2', '3'],
        'age': [25, 35, 45],
        'income': [50000, 75000, 100000],
        'gender': ['Female', 'Male', 'Other'],
        'sentiment_score': [3.5, 4.2, 2.8],
        'avg_spend': [150, 300, 200],
        'total_spend': [1500, 3000, 2000],
        'fav_category': ['Electronics', 'Clothing', 'Groceries'],
        'content': ['Great service', 'Not happy', 'Average'],
        'interests': ['Tech', 'Sports', 'Food']
    })

def test_load_data(sample_data):
    with patch('pandas.read_csv') as mock_read:
        mock_read.side_effect = [
            sample_data,  # customer_profiles.csv
            pd.DataFrame({
                'customer_id': ['1', '2', '3'], 
                'sentiment_score': [3.5, 4.2, 2.8],
                'content': ['a', 'b', 'c']
            }),  # social_media.csv
            pd.DataFrame({
                'customer_id': ['1', '2', '3'], 
                'amount': [150, 300, 200],
                'category': ['A', 'B', 'C']
            })  # transactions.csv
        ]
        
        with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode:
            mock_encode.return_value = np.random.rand(3, 768).tolist()
            
            result = load_data()
            assert isinstance(result, pd.DataFrame)
            assert 'embedding' in result.columns
            assert result['customer_id'].dtype == 'object'  # Check string type

def test_check_bias(sample_data):
    metrics = check_bias(sample_data)
    assert isinstance(metrics, dict)
    assert 'gender_impact' in metrics
    assert 'income_fairness' in metrics
    assert np.isnan(metrics['gender_impact']) or 0 <= metrics['gender_impact'] <= 2
    assert np.isnan(metrics['income_fairness']) or -1 <= metrics['income_fairness'] <= 1

def test_run_benchmarking(sample_data):
    sample_data['embedding'] = [[0.1]*768]*3
    benchmarks = run_benchmarking(sample_data)
    
    assert isinstance(benchmarks, dict)
    required_keys = ['cf_rmse', 'xgb_mae', 'baseline_mae', 'improvement_pct', 'variance_explained']
    assert all(key in benchmarks for key in required_keys)
    assert all(not np.isnan(v) for v in benchmarks.values())

def test_generate_recommendation():
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{'generated_text': "Test output [/INST] Final recommendation"}]
    
    customer_data = {
        'age': 35,
        'income': 75000,
        'avg_spend': 300,
        'interests': 'Tech, Travel',
        'sentiment_score': 4.2
    }
    
    result = generate_recommendation(mock_pipe, customer_data)
    assert result == "Final recommendation"
    assert mock_pipe.call_count == 1