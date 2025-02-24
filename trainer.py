import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import datetime
import joblib

class FoodPreferencePredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None
        self.target_encoder = None
        
    def preprocess_features(self, df, training=True):
        # Initialize storage for encoded data
        encoded_data = {}
        
        # Process time feature
        df['hour'] = df['time'].apply(lambda x: int(x.split(':')[0]))
        df['minute'] = df['time'].apply(lambda x: int(x.split(':')[1]))
        
        # Categorical features to encode
        categorical_features = ['state', 'location', 'Diet', 'weather', 
                              'marital status', 'past preferences', 
                              'preference 1', 'preference 2']
        
        # Numerical features
        numerical_features = ['age', 'hour', 'minute']
        
        # Encode categorical features
        for feature in categorical_features:
            if training:
                # Create new encoder for each feature during training
                encoder = LabelEncoder()
                encoded_data[feature] = encoder.fit_transform(df[feature])
                self.label_encoders[feature] = encoder
            else:
                # Use existing encoder during prediction
                encoder = self.label_encoders[feature]
                encoded_data[feature] = encoder.transform(df[feature])
        
        # Add numerical features
        for feature in numerical_features:
            encoded_data[feature] = df[feature]
        
        # Convert to DataFrame
        encoded_df = pd.DataFrame(encoded_data)
        
        # Scale numerical features
        if training:
            numerical_scaled = self.scaler.fit_transform(encoded_df[numerical_features])
        else:
            numerical_scaled = self.scaler.transform(encoded_df[numerical_features])
            
        encoded_df[numerical_features] = numerical_scaled
        
        if training:
            self.feature_names = encoded_df.columns
            
        return encoded_df
    
    def train(self, df):
        # Preprocess features
        X = self.preprocess_features(df, training=True)
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(df['prefered food item'])
        
        # Train model
        self.model.fit(X, y)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict(self, user_data):
        # Preprocess user data
        X = self.preprocess_features(user_data, training=False)
        
        # Make prediction
        prediction_encoded = self.model.predict(X)
        
        # Decode prediction
        prediction = self.target_encoder.inverse_transform(prediction_encoded)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        
        # Get top 3 predictions with probabilities
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_3_predictions = self.target_encoder.inverse_transform(top_3_indices)
        top_3_probabilities = probabilities[0][top_3_indices]
        
        return prediction[0], list(zip(top_3_predictions, top_3_probabilities))
    
    def save_model(self, filename):
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder
        }
        joblib.dump(model_data, filename)
    
    @classmethod
    def load_model(cls, filename):
        predictor = cls()
        model_data = joblib.load(filename)
        predictor.model = model_data['model']
        predictor.label_encoders = model_data['label_encoders']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.target_encoder = model_data['target_encoder']
        return predictor

# Example usage
def main():
    # Load the dataset
    df = pd.read_csv('indian_food_preferences.csv')
    
    # Create and train the model
    predictor = FoodPreferencePredictor()
    feature_importance = predictor.train(df)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Save the model
    predictor.save_model('food_preference_model.joblib')
    
    # Example prediction
    example_user = pd.DataFrame({
        'age': [30],
        'state': ['Maharashtra'],
        'location': ['Mumbai'],
        'Diet': ['Vegetarian'],
        'weather': ['Hot'],
        'marital status': ['Single'],
        'past preferences': ['South Indian'],
        'preference 1': ['Spicy'],
        'preference 2': ['Traditional'],
        'time': ['14:30']
    })
    
    prediction, top_3 = predictor.predict(example_user)
    
    print("\nPredicted food preference:", prediction)
    print("\nTop 3 predictions with probabilities:")
    for food, prob in top_3:
        print(f"{food}: {prob:.2%}")

    # Create vector representation
    def get_prediction_vector(predictor, user_data):
        # Get all possible food items from target encoder
        all_foods = predictor.target_encoder.classes_
        
        # Get prediction probabilities
        X = predictor.preprocess_features(user_data, training=False)
        probabilities = predictor.model.predict_proba(X)[0]
        
        # Create dictionary mapping food items to their probabilities
        food_vector = dict(zip(all_foods, probabilities))
        return food_vector

    # Get vector representation
    prediction_vector = get_prediction_vector(predictor, example_user)
    print("\nVector representation of prediction (showing top 5 by probability):")
    sorted_items = sorted(prediction_vector.items(), key=lambda x: x[1], reverse=True)[:5]
    for food, prob in sorted_items:
        print(f"{food}: {prob:.4f}")

if __name__ == "__main__":
    main()