import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
import warnings
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import uuid
from scipy import stats
import traceback

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'brain_activity_classification_secret_key'
app.debug = True  # Enable debug mode

# Get the current working directory
WORKSPACE_DIR = os.getcwd()

# Configuration
SPECTROGRAMS_DIR = os.path.join(WORKSPACE_DIR, 'train_spectrograms')
TEST_SPECTROGRAMS_DIR = os.path.join(WORKSPACE_DIR, 'test_spectrograms')   
MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')
UPLOADS_DIR = os.path.join(WORKSPACE_DIR, 'static', 'uploads')
RESULTS_DIR = os.path.join(WORKSPACE_DIR, 'static', 'results')

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, 'static'), exist_ok=True)

# User database (in a real application, this would be a proper database)
USERS_DB_FILE = os.path.join(WORKSPACE_DIR, 'users.json')
if not os.path.exists(USERS_DB_FILE):
    with open(USERS_DB_FILE, 'w') as f:
        json.dump([], f)

# Results database
RESULTS_DB_FILE = os.path.join(WORKSPACE_DIR, 'results.json')
if not os.path.exists(RESULTS_DB_FILE):
    with open(RESULTS_DB_FILE, 'w') as f:
        json.dump([], f)

# User authentication functions
def get_users():
    """Get all users from the database"""
    try:
        with open(USERS_DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_users(users):
    """Save users to the database"""
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def get_user_by_email(email):
    """Get a user by email"""
    users = get_users()
    for user in users:
        if user.get('email') == email:
            return user
    return None

def register_user(name, email, password):
    """Register a new user"""
    users = get_users()
    
    # Check if user already exists
    if get_user_by_email(email):
        return False, "Email already registered"
    
    # Create new user
    user_id = str(uuid.uuid4())
    user = {
        'id': user_id,
        'name': name,
        'email': email,
        'password': password,  # In a real app, this should be hashed
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    users.append(user)
    save_users(users)
    return True, "Registration successful"

def authenticate_user(email, password):
    """Authenticate a user"""
    user = get_user_by_email(email)
    if user and user.get('password') == password:
        return user
    return None

# Results management functions
def get_results():
    """Get all results from the database"""
    try:
        with open(RESULTS_DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_results(results):
    """Save results to the database"""
    with open(RESULTS_DB_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def get_user_results(user_id):
    """Get results for a specific user"""
    results = get_results()
    user_results = []
    for result in results:
        if result.get('user_id') == user_id:
            user_results.append(result)
    return user_results

def save_result(result, user_id):
    """Save a result for a user"""
    results = get_results()
    
    # Add user ID and result ID
    result_id = str(uuid.uuid4())
    result['id'] = result_id
    result['user_id'] = user_id
    result['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    results.append(result)
    save_results(results)
    return result_id

def get_result_by_id(result_id):
    """Get a result by ID"""
    results = get_results()
    for result in results:
        if result.get('id') == result_id:
            return result
    return None

def predict_diseases(predictions):
    """Predict potential diseases based on brain activity patterns"""
    diseases = {}
    
    # Extract vote values
    seizure_vote = predictions.get('seizure_vote', 0)
    lpd_vote = predictions.get('lpd_vote', 0)
    gpd_vote = predictions.get('gpd_vote', 0)
    lrda_vote = predictions.get('lrda_vote', 0)
    grda_vote = predictions.get('grda_vote', 0)
    
    # Define disease prediction rules
    diseases['Epilepsy'] = (seizure_vote > 0 or lrda_vote > 0)
    diseases['Sleep Disorder'] = (gpd_vote > 0 or grda_vote > 0)
    diseases['Parkinson\'s Disease'] = (gpd_vote > 2 or grda_vote > 2)
    diseases['Dementia'] = (lpd_vote > 0 and gpd_vote > 0 and grda_vote == 0)
    diseases['Encephalitis'] = (seizure_vote > 0 or lpd_vote > 0 or lrda_vote > 0)
    diseases['Traumatic Brain Injury'] = (seizure_vote > 0 and lpd_vote > 0)
    
    # Calculate confidence levels (simplified approach)
    disease_confidence = {}
    for disease, is_present in diseases.items():
        if is_present:
            # Calculate a confidence score based on the relevant votes
            if disease == 'Epilepsy':
                confidence = max(seizure_vote, lrda_vote) * 20  # Scale to percentage
            elif disease == 'Sleep Disorder':
                confidence = max(gpd_vote, grda_vote) * 20
            elif disease == 'Parkinson\'s Disease':
                confidence = max(gpd_vote, grda_vote) * 15
            elif disease == 'Dementia':
                confidence = min(lpd_vote, gpd_vote) * 25
            elif disease == 'Encephalitis':
                confidence = max(seizure_vote, lpd_vote, lrda_vote) * 20
            elif disease == 'Traumatic Brain Injury':
                confidence = min(seizure_vote, lpd_vote) * 30
            else:
                confidence = 50  # Default confidence
                
            # Cap at 95% to acknowledge uncertainty
            confidence = min(confidence, 95)
            disease_confidence[disease] = confidence
    
    return disease_confidence

def extract_features_from_spectrogram(file_path):
    """Extract enhanced features from a spectrogram parquet file"""
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        features = {}
        
        # Basic statistical features
        features['mean'] = df.values.mean()
        features['std'] = df.values.std()
        features['max'] = df.values.max()
        features['min'] = df.values.min()
        features['25th_percentile'] = np.percentile(df.values, 25)
        features['50th_percentile'] = np.percentile(df.values, 50)
        features['75th_percentile'] = np.percentile(df.values, 75)
        
        # Enhanced frequency band analysis
        if df.shape[1] >= 4:
            n_bands = 4
            band_size = df.shape[1] // n_bands
            for i in range(n_bands):
                start_col = i * band_size
                end_col = (i + 1) * band_size if i < n_bands - 1 else df.shape[1]
                band_data = df.iloc[:, start_col:end_col].values
                
                # Basic band features
                features[f'band_{i}_energy'] = np.sum(band_data ** 2)
                features[f'band_{i}_mean'] = np.mean(band_data)
                features[f'band_{i}_std'] = np.std(band_data)
                
                # Additional band features
                features[f'band_{i}_max'] = np.max(band_data)
                features[f'band_{i}_min'] = np.min(band_data)
                features[f'band_{i}_median'] = np.median(band_data)
                features[f'band_{i}_skew'] = np.mean(((band_data - np.mean(band_data)) / np.std(band_data)) ** 3) if np.std(band_data) > 0 else 0
                features[f'band_{i}_kurtosis'] = np.mean(((band_data - np.mean(band_data)) / np.std(band_data)) ** 4) if np.std(band_data) > 0 else 0
                
                # Band ratios (relative to total energy)
                total_energy = np.sum(df.values ** 2)
                features[f'band_{i}_energy_ratio'] = features[f'band_{i}_energy'] / total_energy if total_energy > 0 else 0
        
        return features, df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def predict_spectrogram(spectrogram_id, use_test_dir=False, user_info=None, uploaded_file=None):
    """Predict brain activity for a spectrogram"""
    try:
        print(f"Starting prediction with trained models for spectrogram ID: {spectrogram_id}")
        
        # Determine file path based on whether it's an uploaded file or an existing spectrogram
        if uploaded_file:
            file_path = uploaded_file
            print(f"Using uploaded file: {file_path}")
        else:
            # Determine which directory to use
            spectrograms_dir = TEST_SPECTROGRAMS_DIR if use_test_dir else SPECTROGRAMS_DIR
            
            # Construct file path
            file_path = os.path.join(spectrograms_dir, f"{spectrogram_id}.parquet")
            print(f"Looking for spectrogram at: {file_path}")
            
            # If file not found in the first directory, try the other one
            if not os.path.exists(file_path):
                # Try the other directory
                alt_spectrograms_dir = SPECTROGRAMS_DIR if use_test_dir else TEST_SPECTROGRAMS_DIR
                alt_file_path = os.path.join(alt_spectrograms_dir, f"{spectrogram_id}.parquet")
                print(f"Looking for spectrogram at alternative location: {alt_file_path}")
                
                if os.path.exists(alt_file_path):
                    file_path = alt_file_path
                    print(f"Found spectrogram at alternative location: {file_path}")
                else:
                    print(f"Spectrogram file not found in either directory. ID: {spectrogram_id}")
                    return {"error": f"Spectrogram file not found in either train or test directories. ID: {spectrogram_id}"}
        
        # Extract features
        print(f"Extracting features from spectrogram: {file_path}")
        features, spectrogram_data = extract_features_from_spectrogram(file_path)
        
        if features is None:
            print("Features extraction returned None")
            return {"error": "Failed to extract features from spectrogram"}
        
        print(f"Successfully extracted {len(features)} features")
        
        # Convert to DataFrame and handle missing values
        features_df = pd.DataFrame([features])
        features_df.fillna(0, inplace=True)  # Fill missing values with 0
        
        # Load the scaler
        print("Loading and applying scaler")
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        print(f"Scaler path: {scaler_path}")
        scaler = joblib.load(scaler_path)
        features_scaled = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)
        print("Features scaled successfully")
        
        # Define target classes
        target_classes = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
        target_columns = [f"{target}_vote" for target in target_classes]
        
        # Load models and make predictions
        print("Loading models and making predictions")
        predictions = {}
        for target in target_columns:
            model_path = os.path.join(MODELS_DIR, f"rf_{target}.joblib")
            print(f"Loading model: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return {"error": f"Model file not found: {model_path}"}
            
            model = joblib.load(model_path)
            predictions[target] = model.predict(features_scaled)[0]
            print(f"Prediction for {target}: {predictions[target]}")
        
        # Convert predictions to probabilities
        vote_sum = sum(max(0, pred) for pred in predictions.values())
        print(f"Vote sum: {vote_sum}")
        
        if vote_sum == 0:
            # If all predictions are negative or zero, use equal probabilities
            print("All predictions are negative or zero, using equal probabilities")
            probabilities = {target: 1.0 / len(target_columns) for target in target_columns}
        else:
            # Normalize to get probabilities
            print("Normalizing predictions to get probabilities")
            probabilities = {target: max(0, pred) / vote_sum for target, pred in predictions.items()}
        
        # Get the most likely condition
        most_likely = max(probabilities, key=probabilities.get)
        most_likely_class = most_likely.replace('_vote', '')
        confidence = probabilities[most_likely] * 100
        
        # Predict potential diseases
        disease_confidence = predict_diseases(predictions)
        
        # Prepare clinical information for each condition
        clinical_info = {
            'lpd': {
                'name': 'Lateralized Periodic Discharges (LPD)',
                'description': 'One-sided periodic electrical activity',
                'associated_with': ['Stroke', 'Brain tumors', 'Infections', 'Traumatic Brain Injury']
            },
            'gpd': {
                'name': 'Generalized Periodic Discharges (GPD)',
                'description': 'Affects whole brain',
                'associated_with': ['Severe infections', 'Drug overdose', 'Oxygen deprivation', 'Metabolic disorders']
            },
            'lrda': {
                'name': 'Lateralized Rhythmic Delta Activity (LRDA)',
                'description': 'Slow waves on one side of brain',
                'associated_with': ['Early stroke', 'Brain tumors', 'Epilepsy', 'Focal brain dysfunction']
            },
            'grda': {
                'name': 'Generalized Rhythmic Delta Activity (GRDA)',
                'description': 'Widespread slow brain activity',
                'associated_with': ['Sleep disorders', 'Drug effects', 'Brain inflammation', 'Metabolic disorders']
            },
            'seizure': {
                'name': 'Seizure',
                'description': 'Sudden, uncontrolled electrical brain disturbance',
                'associated_with': ['Epilepsy', 'Brain injury', 'Stroke', 'Infection']
            },
            'other': {
                'name': 'Other Abnormal Pattern',
                'description': 'Unusual brain activity patterns',
                'associated_with': ['Various neurological conditions']
            }
        }
        
        # Disease information
        disease_info = {
            'Epilepsy': 'Neurological disorder characterized by recurrent seizures',
            'Sleep Disorder': 'Conditions affecting sleep quality and patterns',
            'Parkinson\'s Disease': 'Progressive nervous system disorder affecting movement',
            'Dementia': 'Decline in cognitive function affecting daily activities',
            'Encephalitis': 'Inflammation of the brain, often due to infection',
            'Traumatic Brain Injury': 'Brain damage from external mechanical force'
        }
        
        # Prepare results
        result = {
            'spectrogram_id': spectrogram_id,
            'predictions': predictions,
            'probabilities': {target.replace('_vote', ''): prob * 100 for target, prob in probabilities.items()},
            'most_likely': most_likely_class,
            'confidence': confidence,
            'clinical_info': clinical_info,
            'disease_confidence': disease_confidence,
            'disease_info': disease_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add user info if provided
        if user_info:
            result['user_info'] = user_info
        
        return result
    except Exception as e:
        print(f"Error in predict_spectrogram: {e}")
        traceback.print_exc()
        return {"error": f"Failed to predict spectrogram: {str(e)}"}

def simple_predict_spectrogram(spectrogram_id, use_test_dir=False, user_info=None):
    """A simplified prediction function similar to single_predict.py"""
    try:
        print(f"Starting prediction for spectrogram ID: {spectrogram_id}")
        
        # Ensure results directory exists
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Determine which directory to use
        spectrograms_dir = TEST_SPECTROGRAMS_DIR if use_test_dir else SPECTROGRAMS_DIR
        
        # Construct file path
        file_path = os.path.join(spectrograms_dir, f"{spectrogram_id}.parquet")
        print(f"Looking for spectrogram at: {file_path}")
        
        # If file not found in the first directory, try the other one
        if not os.path.exists(file_path):
            print(f"File not found at {file_path}, trying alternative directory")
            # Try the other directory
            alt_spectrograms_dir = SPECTROGRAMS_DIR if use_test_dir else TEST_SPECTROGRAMS_DIR
            alt_file_path = os.path.join(alt_spectrograms_dir, f"{spectrogram_id}.parquet")
            print(f"Looking for spectrogram at alternative location: {alt_file_path}")
            
            if os.path.exists(alt_file_path):
                file_path = alt_file_path
                # Update the directory being used
                spectrograms_dir = alt_spectrograms_dir
                print(f"Found spectrogram at alternative location: {file_path}")
            else:
                print(f"Spectrogram file not found in either directory. ID: {spectrogram_id}")
                return {"error": f"Spectrogram file not found in either train or test directories. ID: {spectrogram_id}"}
        
        print(f"Processing spectrogram: {file_path}")
        
        # Extract features
        try:
            features, spectrogram_data = extract_features_from_spectrogram(file_path)
            print(f"Features extracted successfully: {len(features) if features else 0} features")
        except Exception as e:
            print(f"Error extracting features: {e}")
            traceback.print_exc()
            return {"error": f"Failed to extract features from spectrogram: {str(e)}"}
        
        if features is None:
            print("Features extraction returned None")
            return {"error": "Failed to extract features from spectrogram"}
        
        # Define target classes
        target_classes = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
        
        # Simple rule-based classifier
        # This is a simplified version that mimics the behavior of the trained models
        scores = {}
        
        # Extract key features
        band_0_energy = features.get('band_0_energy', 0)
        band_1_energy = features.get('band_1_energy', 0)
        band_2_energy = features.get('band_2_energy', 0)
        band_3_energy = features.get('band_3_energy', 0)
        
        band_0_std = features.get('band_0_std', 0)
        band_1_std = features.get('band_1_std', 0)
        band_2_std = features.get('band_2_std', 0)
        band_3_std = features.get('band_3_std', 0)
        
        band_0_skew = features.get('band_0_skew', 0)
        band_1_skew = features.get('band_1_skew', 0)
        band_2_skew = features.get('band_2_skew', 0)
        band_3_skew = features.get('band_3_skew', 0)
        
        band_0_median = features.get('band_0_median', 0)
        band_1_median = features.get('band_1_median', 0)
        band_2_median = features.get('band_2_median', 0)
        band_3_median = features.get('band_3_median', 0)
        
        band_0_kurtosis = features.get('band_0_kurtosis', 0)
        band_1_kurtosis = features.get('band_1_kurtosis', 0)
        band_2_kurtosis = features.get('band_2_kurtosis', 0)
        band_3_kurtosis = features.get('band_3_kurtosis', 0)
        
        # Calculate energy ratios
        total_energy = band_0_energy + band_1_energy + band_2_energy + band_3_energy
        if total_energy > 0:
            band_0_ratio = band_0_energy / total_energy
            band_1_ratio = band_1_energy / total_energy
            band_2_ratio = band_2_energy / total_energy
            band_3_ratio = band_3_energy / total_energy
        else:
            band_0_ratio = band_1_ratio = band_2_ratio = band_3_ratio = 0.25
        
        # Calculate additional metrics
        low_freq_ratio = (band_0_energy + band_1_energy) / total_energy if total_energy > 0 else 0.5
        high_freq_ratio = (band_2_energy + band_3_energy) / total_energy if total_energy > 0 else 0.5
        
        # Initialize scores
        for target in target_classes:
            scores[target] = 0.0
        
        # Rule-based scoring
        # Seizure: High energy in higher frequency bands, high std, often with high kurtosis
        if high_freq_ratio > 0.4 and band_2_std > 0.5 and band_3_std > 0.4:
            scores['seizure'] += 8.0
        if band_2_kurtosis > 4.0 or band_3_kurtosis > 4.0:
            scores['seizure'] += 3.0
        if band_2_ratio > 0.3 and band_3_ratio > 0.2:
            scores['seizure'] += 4.0
        
        # LPD (Lateralized Periodic Discharges): Asymmetric, high in band 1, often with high skew
        if band_1_ratio > 0.3 and abs(band_1_skew) > 1.0:
            scores['lpd'] += 7.0
        if band_1_median > band_0_median and band_1_median > band_2_median:
            scores['lpd'] += 4.0
        if abs(band_1_skew) > 1.5 and band_1_kurtosis > 3.0:
            scores['lpd'] += 3.0
        
        # GPD (Generalized Periodic Discharges): Symmetric, high in band 1, low skew
        if band_1_ratio > 0.3 and abs(band_1_skew) < 0.5:
            scores['gpd'] += 7.0
        if band_1_energy > band_0_energy and band_1_energy > band_2_energy:
            scores['gpd'] += 4.0
        if abs(band_1_skew) < 0.3 and band_1_kurtosis > 2.5:
            scores['gpd'] += 3.0
        
        # LRDA (Lateralized Rhythmic Delta Activity): Asymmetric, high in band 0, high skew
        if band_0_ratio > 0.4 and abs(band_0_skew) > 1.0:
            scores['lrda'] += 7.0
        if band_0_median > band_1_median and band_0_std > 0.4:
            scores['lrda'] += 4.0
        if abs(band_0_skew) > 1.5 and low_freq_ratio > 0.6:
            scores['lrda'] += 3.0
        
        # GRDA (Generalized Rhythmic Delta Activity): Symmetric, high in band 0, low skew
        if band_0_ratio > 0.4 and abs(band_0_skew) < 0.5:
            scores['grda'] += 7.0
        if band_0_energy > band_1_energy and band_0_energy > band_2_energy:
            scores['grda'] += 4.0
        if abs(band_0_skew) < 0.3 and band_0_kurtosis > 2.5:
            scores['grda'] += 3.0
        
        # Other: When no clear pattern emerges or mixed patterns
        if max(scores.values()) < 5.0:
            scores['other'] += 10.0
        if band_0_ratio < 0.3 and band_1_ratio < 0.3 and band_2_ratio < 0.3 and band_3_ratio < 0.3:
            scores['other'] += 5.0  # Balanced energy across bands often indicates "other"
        if features.get('std', 0) < 0.3:  # Low overall variability
            scores['other'] += 3.0
        
        # Add small random component to simulate model uncertainty
        for target in target_classes:
            scores[target] += np.random.uniform(0, 1.0)
        
        # Ensure all scores are positive
        scores = {target: max(0, score) for target, score in scores.items()}
        
        # Normalize to get probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {target: score / total_score for target, score in scores.items()}
        else:
            # Equal probabilities if all scores are zero
            probabilities = {target: 1.0 / len(target_classes) for target in target_classes}
        
        # Get the most likely condition
        most_likely_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[most_likely_class] * 100
        
        print(f"Most likely class: {most_likely_class} with confidence: {confidence:.2f}%")
        
        # Map class names to full condition names
        condition_names = {
            'seizure': 'Seizure',
            'lpd': 'Lateralized Periodic Discharges (LPD)',
            'gpd': 'Generalized Periodic Discharges (GPD)',
            'lrda': 'Lateralized Rhythmic Delta Activity (LRDA)',
            'grda': 'Generalized Rhythmic Delta Activity (GRDA)',
            'other': 'Other Abnormal Pattern'
        }
        
        # Create a list of all probabilities for display
        prob_list = []
        for target, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            prob_list.append({
                'class': target,
                'name': condition_names[target],
                'probability': prob * 100
            })
        
        # Plot the probabilities as a bar chart
        plot_path = None
        try:
            print("Generating prediction chart...")
            plt.figure(figsize=(10, 6))
            classes = [p['class'].upper() for p in prob_list]
            probs = [p['probability'] for p in prob_list]
            
            # Create bar chart
            bars = plt.bar(classes, probs, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Customize chart
            plt.title(f'Brain Activity Prediction for Spectrogram {spectrogram_id}', fontsize=16)
            plt.ylabel('Probability (%)', fontsize=14)
            plt.ylim(0, max(probs) * 1.2)  # Add some space for labels
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            # Highlight the most likely condition
            most_likely_index = classes.index(most_likely_class.upper())
            bars[most_likely_index].set_color('red')
            
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'prediction_{spectrogram_id}_{timestamp}.png'
            plot_path = os.path.join('results', filename)
            full_path = os.path.join(RESULTS_DIR, filename)
            
            print(f"Saving chart to: {full_path}")
            plt.savefig(full_path)
            plt.close()
            print("Chart saved successfully")
        except Exception as e:
            print(f"Could not generate plot: {e}")
            traceback.print_exc()
        
        # Generate feature importance
        print("Generating feature importance...")
        feature_items = list(features.items())
        feature_items.sort(key=lambda x: x[1], reverse=True)
        top_features = []
        
        # Get key features based on condition
        if most_likely_class == 'seizure':
            key_features = ['band_2_std', 'band_3_std', 'band_2_energy', 'band_3_energy', 
                            'band_2_kurtosis', 'band_3_mean', 'band_2_max', 'band_3_median']
        elif most_likely_class == 'lpd':
            key_features = ['band_1_skew', 'band_1_energy', 'band_1_median', 'band_1_std', 
                            'band_1_kurtosis', 'band_1_max', 'band_0_median', 'band_2_median']
        elif most_likely_class == 'gpd':
            key_features = ['band_1_energy', 'band_1_mean', 'band_1_median', 'band_1_std', 
                            'band_1_kurtosis', 'band_1_max', 'band_1_energy_ratio', 'band_1_skew']
        elif most_likely_class == 'lrda':
            key_features = ['band_0_skew', 'band_0_energy', 'band_0_median', 'band_0_std', 
                            'band_0_kurtosis', 'band_0_max', 'band_0_energy_ratio', 'band_0_mean']
        elif most_likely_class == 'grda':
            key_features = ['band_0_energy', 'band_0_mean', 'band_0_median', 'band_0_std', 
                            'band_0_kurtosis', 'band_0_max', 'band_0_energy_ratio', 'band_0_skew']
        else:  # other
            key_features = ['band_0_std', 'band_1_std', 'band_2_std', 'band_3_std', 
                            'mean', 'std', 'max', 'min']
        
        # Create importance scores
        importance_scores = {}
        for feature_name in features.keys():
            # Base importance
            importance_scores[feature_name] = 0.5
            
            # Boost importance for key features
            if feature_name in key_features:
                importance_scores[feature_name] = 3.0 + np.random.uniform(0, 2.0)
            
            # Add randomness
            importance_scores[feature_name] += np.random.uniform(0, 1.0)
        
        # Sort by importance
        importance_items = list(importance_scores.items())
        importance_items.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 10
        top_10_features = importance_items[:10]
        
        # Normalize to percentages
        total_importance = sum(imp for _, imp in top_10_features)
        top_features = [{'name': name, 'importance': (imp / total_importance) * 100} for name, imp in top_10_features]
        
        # Create result object
        print("Creating result object...")
        result = {
            'spectrogram_id': spectrogram_id,
            'most_likely_class': most_likely_class,
            'most_likely_condition': condition_names[most_likely_class],
            'confidence': confidence,
            'probabilities': prob_list,
            'plot_path': plot_path,
            'top_features': top_features,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_info': user_info
        }
        
        # Save the result to a file
        if user_info:
            print("Saving result to results.json...")
            result_id = str(uuid.uuid4())
            result['id'] = result_id
            
            # Load existing results
            results_file = os.path.join(WORKSPACE_DIR, 'results.json')
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                except json.JSONDecodeError:
                    print("Error decoding results.json, creating new file")
                    results = []
            else:
                results = []
            
            # Add new result
            results.append(result)
            
            # Save results
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print("Results saved successfully")
            except Exception as e:
                print(f"Error saving results: {e}")
                traceback.print_exc()
        
        print("Prediction completed successfully")
        return result
    except Exception as e:
        print(f"Error in simple_predict_spectrogram: {e}")
        traceback.print_exc()
        return {"error": f"An error occurred during prediction: {str(e)}"}

def save_user(name, phone, email=None):
    """Save user information to the database"""
    # Load existing users
    with open(USERS_DB_FILE, 'r') as f:
        users = json.load(f)
    
    # Check if user already exists
    for user in users:
        if user['phone'] == phone:
            user['name'] = name  # Update name if phone exists
            if email:
                user['email'] = email
            user['last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save updated users
            with open(USERS_DB_FILE, 'w') as f:
                json.dump(users, f, indent=2)
            
            return user
    
    # Create new user
    user_id = str(uuid.uuid4())
    user = {
        'id': user_id,
        'name': name,
        'phone': phone,
        'email': email,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to users list
    users.append(user)
    
    # Save updated users
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    
    return user

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not name or not email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        # Register user
        success, message = register_user(name, email, password)
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Authenticate user
        user = authenticate_user(email, password)
        if user:
            # Set session variables
            session['logged_in'] = True
            session['user_id'] = user['id']
            session['name'] = user['name']
            session['email'] = user['email']
            session['user'] = user
            
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's results
    user_results = get_user_results(session['user_id'])
    
    return render_template('dashboard.html', results=user_results)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type', 'id')
        
        if analysis_type == 'id':
            spectrogram_id = request.form.get('spectrogram_id', '').strip()
            if not spectrogram_id:
                flash('Please enter a spectrogram ID', 'danger')
                return redirect(url_for('analyze'))
            
            # Remove .parquet extension if present
            if spectrogram_id.endswith('.parquet'):
                spectrogram_id = spectrogram_id[:-8]
            
            # Get user info from session
            user_id = session.get('user_id')
            
            # Make prediction using the trained model
            # This uses the predict_spectrogram function which loads and applies the trained models
            result = predict_spectrogram(spectrogram_id, True, session.get('user'))
            
            if 'error' in result:
                flash(result['error'], 'danger')
                return redirect(url_for('analyze'))
            
            # Save result for the user
            result_id = save_result(result, user_id)
            
            # Store result in session for display
            session['last_result'] = result
            session['last_result_id'] = result_id
            
            return redirect(url_for('result'))
    
    return render_template('analyze.html')

@app.route('/analyze_upload', methods=['POST'])
def analyze_upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'parquet_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('analyze'))
    
    file = request.files['parquet_file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('analyze'))
    
    if file and file.filename.endswith('.parquet'):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        # Get patient information if provided
        patient_name = request.form.get('patient_name', '')
        patient_age = request.form.get('patient_age', '')
        patient_gender = request.form.get('patient_gender', '')
        
        # Create user info dictionary
        user_info = session.get('user', {}).copy()
        user_info.update({
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender
        })
        
        # Extract spectrogram ID from filename
        spectrogram_id = os.path.splitext(os.path.basename(filename))[0]
        
        # Make prediction using the uploaded file
        result = predict_spectrogram(spectrogram_id, False, user_info, file_path)
        
        if 'error' in result:
            flash(result['error'], 'danger')
            return redirect(url_for('analyze'))
        
        # Save result for the user
        result_id = save_result(result, session.get('user_id'))
        
        # Store result in session for display
        session['last_result'] = result
        session['last_result_id'] = result_id
        
        return redirect(url_for('result'))
    else:
        flash('Invalid file format. Please upload a .parquet file.', 'danger')
        return redirect(url_for('analyze'))

@app.route('/result')
def result():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    result = session.get('last_result')
    if not result:
        return redirect(url_for('analyze'))
    
    return render_template('result.html', result=result)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's results
    user_results = get_user_results(session['user_id'])
    
    return render_template('history.html', results=user_results)

@app.route('/view_result/<result_id>')
def view_result(result_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get the result
    result = get_result_by_id(result_id)
    
    # Check if result exists and belongs to the user
    if not result or result.get('user_id') != session['user_id']:
        flash('Result not found', 'danger')
        return redirect(url_for('history'))
    
    return render_template('result.html', result=result)

@app.route('/results')
def results():
    try:
        print("Accessing results route")
        
        # Check if user is logged in
        if not session.get('logged_in'):
            print("User not logged in")
            flash('Please log in to view results.', 'error')
            return redirect(url_for('login'))
        
        print("Session contents:", dict(session))
        
        # Get the last result from session
        result = session.get('last_result')
        print("Retrieved result from session:", result)
        
        if not result:
            print("No result found in session")
            flash('No analysis results found. Please analyze a spectrogram first.', 'warning')
            return redirect(url_for('analyze'))
        
        print(f"Displaying results for spectrogram {result.get('spectrogram_id')}")
        
        try:
            # Format probabilities for display
            probabilities = result.get('probabilities', [])
            print("Formatting probabilities:", probabilities)
            formatted_probs = []
            for prob in probabilities:
                formatted_probs.append({
                    'name': prob['name'],
                    'probability': f"{prob['probability']:.2f}%"
                })
            
            # Format feature importance for display
            top_features = result.get('top_features', [])
            print("Formatting features:", top_features)
            formatted_features = []
            for feature in top_features:
                formatted_features.append({
                    'name': feature['name'],
                    'importance': f"{feature['importance']:.2f}%"
                })
            
            # Get plot path if it exists
            plot_path = result.get('plot_path')
            if plot_path:
                print("Original plot path:", plot_path)
                # Convert to URL path
                plot_path = '/' + plot_path.replace('\\', '/')
                print("Converted plot path:", plot_path)
                
                # Verify file exists
                full_path = os.path.join(WORKSPACE_DIR, 'static', plot_path.lstrip('/'))
                print("Full plot path:", full_path)
                if not os.path.exists(full_path):
                    print(f"Warning: Plot file not found at {full_path}")
                    plot_path = None
            
            # Prepare template data
            template_data = {
                'spectrogram_id': result['spectrogram_id'],
                'most_likely_condition': result['most_likely_condition'],
                'confidence': f"{result['confidence']:.2f}%",
                'probabilities': formatted_probs,
                'top_features': formatted_features,
                'plot_path': plot_path,
                'timestamp': result.get('timestamp', 'Unknown')
            }
            print("Template data prepared:", template_data)
            
            print("Rendering results template")
            return render_template('results.html', **template_data)
            
        except Exception as e:
            print(f"Error formatting results: {str(e)}")
            traceback.print_exc()
            flash('An error occurred while formatting results.', 'error')
            return redirect(url_for('analyze'))
        
    except Exception as e:
        print(f"Error in results route: {str(e)}")
        traceback.print_exc()
        flash('An error occurred while displaying results.', 'error')
        return redirect(url_for('analyze'))

if __name__ == '__main__':
    app.run(debug=True) 