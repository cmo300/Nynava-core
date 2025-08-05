"""
Nynava - Main Flask Application
Decentralized Healthcare AI Platform
"""

import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import structlog
from dotenv import load_dotenv

# Import custom modules
from anonymization import DataAnonymizer
from blockchain import ConsentManager
from federated_learning import FederatedLearningCoordinator
from models.ai_models import MedicalAIModels
from utils.ipfs_client import IPFSClient
from utils.database import DatabaseManager
from utils.auth import AuthManager
from utils.gamification import GamificationManager

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    # Enable CORS
    CORS(app, origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(','))
    
    # Proxy fix for deployment
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Initialize components
    anonymizer = DataAnonymizer()
    consent_manager = ConsentManager()
    fl_coordinator = FederatedLearningCoordinator()
    ai_models = MedicalAIModels()
    ipfs_client = IPFSClient()
    db_manager = DatabaseManager()
    auth_manager = AuthManager()
    gamification = GamificationManager()
    
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'nynava-backend',
            'version': '1.0.0'
        })
    
    # Authentication Routes
    @app.route('/api/v1/auth/register', methods=['POST'])
    def register():
        """User registration"""
        try:
            data = request.get_json()
            result = auth_manager.register_user(
                email=data['email'],
                password=data['password'],
                role=data.get('role', 'patient')
            )
            return jsonify(result), 201
        except Exception as e:
            logger.error("Registration failed", error=str(e))
            return jsonify({'error': 'Registration failed'}), 400
    
    @app.route('/api/v1/auth/login', methods=['POST'])
    def login():
        """User login"""
        try:
            data = request.get_json()
            result = auth_manager.authenticate_user(
                email=data['email'],
                password=data['password']
            )
            return jsonify(result)
        except Exception as e:
            logger.error("Login failed", error=str(e))
            return jsonify({'error': 'Invalid credentials'}), 401
    
    # Consent Management Routes
    @app.route('/api/v1/consent/submit', methods=['POST'])
    def submit_consent():
        """Submit patient consent"""
        try:
            data = request.get_json()
            user_id = auth_manager.get_current_user_id(request)
            
            consent_hash = consent_manager.record_consent(
                user_id=user_id,
                consent_data=data,
                blockchain_record=True
            )
            
            # Award points for consent submission
            gamification.award_points(user_id, 'consent_submission', 50)
            
            return jsonify({
                'success': True,
                'consent_hash': consent_hash,
                'message': 'Consent recorded successfully'
            })
        except Exception as e:
            logger.error("Consent submission failed", error=str(e))
            return jsonify({'error': 'Consent submission failed'}), 400
    
    @app.route('/api/v1/consent/status/<user_id>')
    def get_consent_status(user_id):
        """Get user consent status"""
        try:
            status = consent_manager.get_consent_status(user_id)
            return jsonify(status)
        except Exception as e:
            logger.error("Failed to get consent status", error=str(e))
            return jsonify({'error': 'Failed to retrieve consent status'}), 400
    
    # Data Upload Routes
    @app.route('/api/v1/data/upload', methods=['POST'])
    def upload_data():
        """Upload patient data with anonymization"""
        try:
            user_id = auth_manager.get_current_user_id(request)
            
            # Check consent status
            if not consent_manager.has_valid_consent(user_id):
                return jsonify({'error': 'Valid consent required'}), 403
            
            files = request.files.getlist('files')
            metadata = request.form.get('metadata', '{}')
            
            results = []
            for file in files:
                # Anonymize data
                anonymized_data = anonymizer.anonymize_file(file)
                
                # Store in IPFS
                ipfs_hash = ipfs_client.add_file(anonymized_data)
                
                # Record in database
                record_id = db_manager.create_data_record(
                    user_id=user_id,
                    ipfs_hash=ipfs_hash,
                    metadata=metadata,
                    file_type=file.content_type
                )
                
                results.append({
                    'record_id': record_id,
                    'ipfs_hash': ipfs_hash,
                    'filename': file.filename
                })
            
            # Award points for data contribution
            gamification.award_points(user_id, 'data_upload', len(files) * 100)
            
            return jsonify({
                'success': True,
                'uploads': results,
                'message': f'Successfully uploaded {len(files)} files'
            })
            
        except Exception as e:
            logger.error("Data upload failed", error=str(e))
            return jsonify({'error': 'Data upload failed'}), 400
    
    # AI Model Routes
    @app.route('/api/v1/models/available')
    def get_available_models():
        """Get list of available AI models"""
        try:
            models = ai_models.get_available_models()
            return jsonify(models)
        except Exception as e:
            logger.error("Failed to get available models", error=str(e))
            return jsonify({'error': 'Failed to retrieve models'}), 400
    
    @app.route('/api/v1/models/run', methods=['POST'])
    def run_model():
        """Run AI model on user data"""
        try:
            user_id = auth_manager.get_current_user_id(request)
            data = request.get_json()
            
            model_name = data['model_name']
            data_records = data['data_records']
            
            # Run federated learning
            results = fl_coordinator.run_inference(
                model_name=model_name,
                user_id=user_id,
                data_records=data_records
            )
            
            # Award points for model usage
            gamification.award_points(user_id, 'model_usage', 25)
            
            return jsonify({
                'success': True,
                'results': results,
                'model_name': model_name
            })
            
        except Exception as e:
            logger.error("Model execution failed", error=str(e))
            return jsonify({'error': 'Model execution failed'}), 400
    
    # Federated Learning Routes
    @app.route('/api/v1/federated/start-training', methods=['POST'])
    def start_federated_training():
        """Start federated learning training session"""
        try:
            data = request.get_json()
            
            session_id = fl_coordinator.start_training_session(
                model_name=data['model_name'],
                participants=data.get('participants', []),
                rounds=data.get('rounds', 10)
            )
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Federated learning session started'
            })
            
        except Exception as e:
            logger.error("Failed to start federated training", error=str(e))
            return jsonify({'error': 'Failed to start training'}), 400
    
    @app.route('/api/v1/federated/status/<session_id>')
    def get_training_status(session_id):
        """Get federated learning training status"""
        try:
            status = fl_coordinator.get_training_status(session_id)
            return jsonify(status)
        except Exception as e:
            logger.error("Failed to get training status", error=str(e))
            return jsonify({'error': 'Failed to retrieve status'}), 400
    
    # Gamification Routes
    @app.route('/api/v1/gamification/profile/<user_id>')
    def get_user_profile(user_id):
        """Get user gamification profile"""
        try:
            profile = gamification.get_user_profile(user_id)
            return jsonify(profile)
        except Exception as e:
            logger.error("Failed to get user profile", error=str(e))
            return jsonify({'error': 'Failed to retrieve profile'}), 400
    
    @app.route('/api/v1/gamification/leaderboard')
    def get_leaderboard():
        """Get community leaderboard"""
        try:
            leaderboard = gamification.get_leaderboard()
            return jsonify(leaderboard)
        except Exception as e:
            logger.error("Failed to get leaderboard", error=str(e))
            return jsonify({'error': 'Failed to retrieve leaderboard'}), 400
    
    # Dataset Management Routes
    @app.route('/api/v1/datasets/list')
    def list_datasets():
        """List available public datasets"""
        try:
            datasets = db_manager.get_public_datasets()
            return jsonify(datasets)
        except Exception as e:
            logger.error("Failed to list datasets", error=str(e))
            return jsonify({'error': 'Failed to retrieve datasets'}), 400
    
    @app.route('/api/v1/datasets/<category>')
    def get_datasets_by_category(category):
        """Get datasets by category"""
        try:
            datasets = db_manager.get_datasets_by_category(category)
            return jsonify(datasets)
        except Exception as e:
            logger.error("Failed to get datasets by category", error=str(e))
            return jsonify({'error': 'Failed to retrieve datasets'}), 400
    
    # Error Handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error", error=str(error))
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(413)
    def too_large(error):
        return jsonify({'error': 'File too large'}), 413
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info("Starting Nynava backend server", port=port, debug=debug)
    app.run(host='0.0.0.0', port=port, debug=debug)