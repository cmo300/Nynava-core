"""
Nynava - Federated Learning Module
Privacy-preserving distributed machine learning using Flower framework
"""

import os
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import flwr as fl
from flwr.server import ServerConfig, start_server
from flwr.client import Client, ClientApp, start_client
from flwr.common import Parameters, FitRes, EvaluateRes, Status, Code
import structlog

logger = structlog.get_logger()

class MedicalTextClassifier(nn.Module):
    """Medical text classification model for federated learning"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_classes: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class MedicalImageClassifier(nn.Module):
    """Medical image classification model for federated learning"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming 224x224 input -> 28x28 after pooling
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

class NynavaFederatedClient(Client):
    """Federated learning client for Nynava platform"""
    
    def __init__(self, client_id: str, model_type: str, local_data_path: str):
        self.client_id = client_id
        self.model_type = model_type
        self.local_data_path = local_data_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model based on type
        if model_type == "text_classification":
            self.model = MedicalTextClassifier()
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        elif model_type == "image_classification":
            self.model = MedicalImageClassifier()
        
        self.model.to(self.device)
        logger.info("Federated client initialized", client_id=client_id, model_type=model_type)
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return model parameters as numpy arrays"""
        return [param.cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param).to(self.device)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train model on local data"""
        try:
            self.set_parameters(parameters)
            
            # Load local training data
            train_loader = self._load_local_data("train")
            
            # Training configuration
            epochs = config.get("epochs", 1)
            learning_rate = config.get("learning_rate", 1e-5)
            
            # Train model
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            num_samples = 0
            
            for epoch in range(epochs):
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    if self.model_type == "text_classification":
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                    
                    elif self.model_type == "image_classification":
                        images = batch['images'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_samples += len(labels)
            
            # Return updated parameters and metrics
            updated_parameters = self.get_parameters({})
            metrics = {
                "train_loss": total_loss / num_samples,
                "num_samples": num_samples,
                "client_id": self.client_id
            }
            
            logger.info("Local training completed", 
                       client_id=self.client_id, 
                       loss=metrics["train_loss"],
                       samples=num_samples)
            
            return updated_parameters, num_samples, metrics
            
        except Exception as e:
            logger.error("Local training failed", client_id=self.client_id, error=str(e))
            raise
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model on local data"""
        try:
            self.set_parameters(parameters)
            
            # Load local evaluation data
            eval_loader = self._load_local_data("eval")
            
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            correct = 0
            num_samples = 0
            
            with torch.no_grad():
                for batch in eval_loader:
                    if self.model_type == "text_classification":
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                        
                        predictions = torch.argmax(outputs, dim=1)
                        correct += (predictions == labels).sum().item()
                    
                    elif self.model_type == "image_classification":
                        images = batch['images'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        predictions = torch.argmax(outputs, dim=1)
                        correct += (predictions == labels).sum().item()
                    
                    total_loss += loss.item()
                    num_samples += len(labels)
            
            accuracy = correct / num_samples
            avg_loss = total_loss / num_samples
            
            metrics = {
                "accuracy": accuracy,
                "eval_loss": avg_loss,
                "num_samples": num_samples,
                "client_id": self.client_id
            }
            
            logger.info("Local evaluation completed",
                       client_id=self.client_id,
                       accuracy=accuracy,
                       loss=avg_loss)
            
            return avg_loss, num_samples, metrics
            
        except Exception as e:
            logger.error("Local evaluation failed", client_id=self.client_id, error=str(e))
            raise
    
    def _load_local_data(self, split: str):
        """Load local training/evaluation data"""
        # This would be implemented to load the actual local data
        # For now, return dummy data loader
        # In practice, this would load anonymized patient data
        
        if self.model_type == "text_classification":
            return self._create_dummy_text_loader(split)
        elif self.model_type == "image_classification":
            return self._create_dummy_image_loader(split)
    
    def _create_dummy_text_loader(self, split: str):
        """Create dummy text data loader for testing"""
        # This is a placeholder - in production, load real anonymized data
        import torch.utils.data as data
        
        class DummyTextDataset(data.Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Dummy medical text
                text = "Patient presents with symptoms of chest pain and shortness of breath."
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': torch.randint(0, 2, (1,)).squeeze()
                }
        
        dataset = DummyTextDataset()
        return data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    def _create_dummy_image_loader(self, split: str):
        """Create dummy image data loader for testing"""
        import torch.utils.data as data
        
        class DummyImageDataset(data.Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Dummy medical image (224x224 RGB)
                return {
                    'images': torch.randn(3, 224, 224),
                    'labels': torch.randint(0, 2, (1,)).squeeze()
                }
        
        dataset = DummyImageDataset()
        return data.DataLoader(dataset, batch_size=8, shuffle=True)

class FederatedLearningCoordinator:
    """Coordinates federated learning sessions for Nynava"""
    
    def __init__(self):
        self.active_sessions = {}
        self.server_address = os.getenv("FLOWER_SERVER_ADDRESS", "localhost:8080")
        self.min_clients = int(os.getenv("FL_MIN_CLIENTS", "2"))
        self.rounds = int(os.getenv("FL_ROUNDS", "10"))
        logger.info("FederatedLearningCoordinator initialized")
    
    def start_training_session(self, model_name: str, participants: List[str], rounds: int = None) -> str:
        """Start a new federated learning training session"""
        try:
            session_id = str(uuid.uuid4())
            
            if rounds is None:
                rounds = self.rounds
            
            session_config = {
                'session_id': session_id,
                'model_name': model_name,
                'participants': participants,
                'rounds': rounds,
                'status': 'initializing',
                'created_at': datetime.utcnow().isoformat(),
                'current_round': 0,
                'metrics': []
            }
            
            self.active_sessions[session_id] = session_config
            
            # Start federated learning server asynchronously
            asyncio.create_task(self._run_federated_server(session_id, model_name, rounds))
            
            logger.info("Federated learning session started", 
                       session_id=session_id, 
                       model=model_name,
                       participants=len(participants))
            
            return session_id
            
        except Exception as e:
            logger.error("Failed to start federated learning session", error=str(e))
            raise
    
    async def _run_federated_server(self, session_id: str, model_name: str, rounds: int):
        """Run the federated learning server"""
        try:
            self.active_sessions[session_id]['status'] = 'running'
            
            # Define strategy for federated averaging
            strategy = fl.server.strategy.FedAvg(
                min_fit_clients=self.min_clients,
                min_evaluate_clients=self.min_clients,
                min_available_clients=self.min_clients,
                evaluate_metrics_aggregation_fn=self._aggregate_metrics,
                fit_metrics_aggregation_fn=self._aggregate_metrics
            )
            
            # Server configuration
            config = ServerConfig(num_rounds=rounds)
            
            # Start server (this would be modified for async operation)
            # For now, we'll simulate the training process
            await self._simulate_federated_training(session_id, rounds)
            
        except Exception as e:
            logger.error("Federated server failed", session_id=session_id, error=str(e))
            self.active_sessions[session_id]['status'] = 'failed'
            self.active_sessions[session_id]['error'] = str(e)
    
    async def _simulate_federated_training(self, session_id: str, rounds: int):
        """Simulate federated training process"""
        try:
            for round_num in range(1, rounds + 1):
                self.active_sessions[session_id]['current_round'] = round_num
                self.active_sessions[session_id]['status'] = f'round_{round_num}'
                
                # Simulate training metrics
                round_metrics = {
                    'round': round_num,
                    'train_loss': np.random.uniform(0.5, 1.0) * (1 - round_num/rounds),
                    'train_accuracy': np.random.uniform(0.7, 0.95) * (round_num/rounds + 0.3),
                    'eval_loss': np.random.uniform(0.4, 0.9) * (1 - round_num/rounds),
                    'eval_accuracy': np.random.uniform(0.75, 0.98) * (round_num/rounds + 0.25),
                    'participants': len(self.active_sessions[session_id]['participants']),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.active_sessions[session_id]['metrics'].append(round_metrics)
                
                # Simulate round duration
                await asyncio.sleep(2)
                
                logger.info("Federated learning round completed",
                           session_id=session_id,
                           round=round_num,
                           accuracy=round_metrics['eval_accuracy'])
            
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['completed_at'] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error("Federated training simulation failed", session_id=session_id, error=str(e))
            raise
    
    def _aggregate_metrics(self, metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate metrics from federated clients"""
        if not metrics:
            return {}
        
        # Extract metrics and sample counts
        total_samples = sum(num_samples for num_samples, _ in metrics)
        
        # Weighted average of metrics
        aggregated = {}
        for metric_name in metrics[0][1].keys():
            if isinstance(metrics[0][1][metric_name], (int, float)):
                weighted_sum = sum(
                    num_samples * client_metrics[metric_name]
                    for num_samples, client_metrics in metrics
                )
                aggregated[metric_name] = weighted_sum / total_samples
        
        aggregated['total_samples'] = total_samples
        aggregated['num_clients'] = len(metrics)
        
        return aggregated
    
    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of federated learning session"""
        try:
            if session_id not in self.active_sessions:
                return {'error': 'Session not found'}
            
            session = self.active_sessions[session_id]
            
            # Calculate progress
            progress = 0
            if session['status'] != 'initializing' and session['rounds'] > 0:
                progress = (session['current_round'] / session['rounds']) * 100
            
            status = {
                'session_id': session_id,
                'status': session['status'],
                'progress': progress,
                'current_round': session['current_round'],
                'total_rounds': session['rounds'],
                'participants': len(session['participants']),
                'created_at': session['created_at'],
                'latest_metrics': session['metrics'][-1] if session['metrics'] else None
            }
            
            if session['status'] == 'completed':
                status['completed_at'] = session.get('completed_at')
                status['final_metrics'] = session['metrics'][-1] if session['metrics'] else None
            
            return status
            
        except Exception as e:
            logger.error("Failed to get training status", session_id=session_id, error=str(e))
            return {'error': 'Failed to retrieve status'}
    
    def run_inference(self, model_name: str, user_id: str, data_records: List[str]) -> Dict[str, Any]:
        """Run inference on user data using trained federated model"""
        try:
            # This would load the trained federated model and run inference
            # For now, return simulated results
            
            results = {
                'model_name': model_name,
                'user_id': user_id,
                'predictions': [],
                'confidence_scores': [],
                'processing_time': np.random.uniform(0.5, 2.0),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Simulate predictions for each data record
            for record_id in data_records:
                if 'text' in model_name.lower():
                    # Text classification results
                    prediction = {
                        'record_id': record_id,
                        'prediction': np.random.choice(['positive', 'negative']),
                        'confidence': np.random.uniform(0.7, 0.95),
                        'risk_score': np.random.uniform(0.1, 0.9)
                    }
                elif 'image' in model_name.lower():
                    # Image classification results
                    prediction = {
                        'record_id': record_id,
                        'prediction': np.random.choice(['normal', 'abnormal']),
                        'confidence': np.random.uniform(0.75, 0.98),
                        'abnormality_score': np.random.uniform(0.0, 0.8)
                    }
                else:
                    # Generic prediction
                    prediction = {
                        'record_id': record_id,
                        'prediction': 'processed',
                        'confidence': np.random.uniform(0.8, 0.95),
                        'score': np.random.uniform(0.2, 0.9)
                    }
                
                results['predictions'].append(prediction)
                results['confidence_scores'].append(prediction['confidence'])
            
            # Calculate aggregate metrics
            results['average_confidence'] = np.mean(results['confidence_scores'])
            results['num_records_processed'] = len(data_records)
            
            logger.info("Inference completed",
                       model=model_name,
                       user_id=user_id,
                       records=len(data_records),
                       avg_confidence=results['average_confidence'])
            
            return results
            
        except Exception as e:
            logger.error("Inference failed", model=model_name, user_id=user_id, error=str(e))
            raise
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available federated learning models"""
        return [
            {
                'name': 'medical_text_classifier',
                'type': 'text_classification',
                'description': 'Medical text classification using BioBERT',
                'supported_tasks': ['diagnosis_prediction', 'symptom_analysis'],
                'privacy_level': 'high',
                'status': 'available'
            },
            {
                'name': 'chest_xray_classifier',
                'type': 'image_classification',
                'description': 'Chest X-ray abnormality detection',
                'supported_tasks': ['pneumonia_detection', 'covid_screening'],
                'privacy_level': 'high',
                'status': 'available'
            },
            {
                'name': 'mental_health_predictor',
                'type': 'text_classification',
                'description': 'Mental health risk assessment from text',
                'supported_tasks': ['depression_screening', 'anxiety_detection'],
                'privacy_level': 'high',
                'status': 'available'
            }
        ]