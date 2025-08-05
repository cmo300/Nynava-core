"""
Nynava - Blockchain Consent Management
Hyperledger Fabric-based consent tracking and management
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

# Note: In production, use actual Hyperledger Fabric SDK
# from hfc.fabric import Client as FabricClient

logger = structlog.get_logger()

class ConsentManager:
    """Blockchain-based consent management for HIPAA compliance"""
    
    def __init__(self):
        self.channel_name = os.getenv("HYPERLEDGER_CHANNEL_NAME", "nynava-channel")
        self.chaincode_name = os.getenv("HYPERLEDGER_CHAINCODE_NAME", "consent-management")
        self.peer_url = os.getenv("HYPERLEDGER_PEER_URL", "grpc://localhost:7051")
        self.orderer_url = os.getenv("HYPERLEDGER_ORDERER_URL", "grpc://localhost:7050")
        
        # In-memory storage for demo (replace with actual blockchain in production)
        self.consent_records = {}
        self.consent_history = {}
        
        # Generate RSA key pair for consent signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        logger.info("ConsentManager initialized", channel=self.channel_name)
    
    def record_consent(self, user_id: str, consent_data: Dict[str, Any], blockchain_record: bool = True) -> str:
        """Record patient consent on blockchain"""
        try:
            # Create consent record
            consent_record = {
                'user_id': user_id,
                'consent_id': self._generate_consent_id(user_id),
                'timestamp': datetime.utcnow().isoformat(),
                'consent_data': consent_data,
                'version': '1.0',
                'status': 'active',
                'expiry_date': (datetime.utcnow() + timedelta(days=365)).isoformat(),  # 1 year expiry
                'ip_address': consent_data.get('ip_address', 'unknown'),
                'user_agent': consent_data.get('user_agent', 'unknown')
            }
            
            # Add specific consent fields
            consent_record.update({
                'general_research_consent': consent_data.get('general_research_consent', False),
                'cancer_research_opt_out': consent_data.get('cancer_research_opt_out', False),
                'mental_health_opt_out': consent_data.get('mental_health_opt_out', False),
                'heart_disease_opt_out': consent_data.get('heart_disease_opt_out', False),
                'data_sharing_consent': consent_data.get('data_sharing_consent', True),
                'ai_processing_consent': consent_data.get('ai_processing_consent', True),
                'withdrawal_method': 'email_request'
            })
            
            # Create digital signature
            consent_hash = self._create_consent_hash(consent_record)
            signature = self._sign_consent(consent_hash)
            
            consent_record['consent_hash'] = consent_hash
            consent_record['digital_signature'] = signature
            
            # Store in blockchain (simulated)
            if blockchain_record:
                blockchain_tx_id = self._submit_to_blockchain(consent_record)
                consent_record['blockchain_tx_id'] = blockchain_tx_id
            
            # Store locally for quick access
            self.consent_records[user_id] = consent_record
            
            # Add to history
            if user_id not in self.consent_history:
                self.consent_history[user_id] = []
            self.consent_history[user_id].append(consent_record)
            
            logger.info("Consent recorded", 
                       user_id=user_id, 
                       consent_id=consent_record['consent_id'],
                       blockchain_tx=consent_record.get('blockchain_tx_id'))
            
            return consent_hash
            
        except Exception as e:
            logger.error("Failed to record consent", user_id=user_id, error=str(e))
            raise
    
    def get_consent_status(self, user_id: str) -> Dict[str, Any]:
        """Get current consent status for user"""
        try:
            if user_id not in self.consent_records:
                return {
                    'user_id': user_id,
                    'has_consent': False,
                    'message': 'No consent record found'
                }
            
            consent = self.consent_records[user_id]
            
            # Check if consent is still valid
            expiry_date = datetime.fromisoformat(consent['expiry_date'])
            is_expired = datetime.utcnow() > expiry_date
            is_active = consent['status'] == 'active' and not is_expired
            
            return {
                'user_id': user_id,
                'has_consent': is_active,
                'consent_id': consent['consent_id'],
                'status': consent['status'],
                'granted_date': consent['timestamp'],
                'expiry_date': consent['expiry_date'],
                'is_expired': is_expired,
                'permissions': {
                    'general_research': consent['general_research_consent'],
                    'cancer_research': not consent['cancer_research_opt_out'],
                    'mental_health_research': not consent['mental_health_opt_out'],
                    'heart_disease_research': not consent['heart_disease_opt_out'],
                    'data_sharing': consent['data_sharing_consent'],
                    'ai_processing': consent['ai_processing_consent']
                },
                'blockchain_verified': 'blockchain_tx_id' in consent
            }
            
        except Exception as e:
            logger.error("Failed to get consent status", user_id=user_id, error=str(e))
            return {
                'user_id': user_id,
                'has_consent': False,
                'error': str(e)
            }
    
    def has_valid_consent(self, user_id: str) -> bool:
        """Check if user has valid consent"""
        status = self.get_consent_status(user_id)
        return status.get('has_consent', False)
    
    def withdraw_consent(self, user_id: str, reason: str = "User request") -> bool:
        """Withdraw user consent"""
        try:
            if user_id not in self.consent_records:
                return False
            
            # Create withdrawal record
            withdrawal_record = {
                'user_id': user_id,
                'original_consent_id': self.consent_records[user_id]['consent_id'],
                'withdrawal_id': self._generate_consent_id(user_id, prefix='WD'),
                'timestamp': datetime.utcnow().isoformat(),
                'reason': reason,
                'status': 'withdrawn'
            }
            
            # Update consent status
            self.consent_records[user_id]['status'] = 'withdrawn'
            self.consent_records[user_id]['withdrawal_date'] = withdrawal_record['timestamp']
            self.consent_records[user_id]['withdrawal_reason'] = reason
            
            # Record withdrawal on blockchain
            withdrawal_hash = self._create_consent_hash(withdrawal_record)
            blockchain_tx_id = self._submit_to_blockchain(withdrawal_record)
            
            # Add to history
            self.consent_history[user_id].append(withdrawal_record)
            
            logger.info("Consent withdrawn", 
                       user_id=user_id, 
                       withdrawal_id=withdrawal_record['withdrawal_id'],
                       reason=reason)
            
            return True
            
        except Exception as e:
            logger.error("Failed to withdraw consent", user_id=user_id, error=str(e))
            return False
    
    def update_consent(self, user_id: str, updated_consent: Dict[str, Any]) -> str:
        """Update existing consent with new preferences"""
        try:
            if user_id not in self.consent_records:
                raise ValueError("No existing consent found")
            
            # Create new version of consent
            old_consent = self.consent_records[user_id]
            new_version = float(old_consent['version']) + 0.1
            
            updated_record = old_consent.copy()
            updated_record.update({
                'version': str(new_version),
                'timestamp': datetime.utcnow().isoformat(),
                'previous_version': old_consent['version'],
                'update_reason': 'user_preference_change'
            })
            
            # Update specific consent fields
            for key in ['cancer_research_opt_out', 'mental_health_opt_out', 'heart_disease_opt_out',
                       'data_sharing_consent', 'ai_processing_consent']:
                if key in updated_consent:
                    updated_record[key] = updated_consent[key]
            
            # Create new hash and signature
            consent_hash = self._create_consent_hash(updated_record)
            signature = self._sign_consent(consent_hash)
            
            updated_record['consent_hash'] = consent_hash
            updated_record['digital_signature'] = signature
            
            # Submit to blockchain
            blockchain_tx_id = self._submit_to_blockchain(updated_record)
            updated_record['blockchain_tx_id'] = blockchain_tx_id
            
            # Update records
            self.consent_records[user_id] = updated_record
            self.consent_history[user_id].append(updated_record)
            
            logger.info("Consent updated", 
                       user_id=user_id, 
                       version=new_version,
                       blockchain_tx=blockchain_tx_id)
            
            return consent_hash
            
        except Exception as e:
            logger.error("Failed to update consent", user_id=user_id, error=str(e))
            raise
    
    def get_consent_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get full consent history for user"""
        try:
            if user_id not in self.consent_history:
                return []
            
            # Return history with sensitive data removed
            history = []
            for record in self.consent_history[user_id]:
                public_record = {
                    'consent_id': record.get('consent_id', record.get('withdrawal_id')),
                    'timestamp': record['timestamp'],
                    'version': record.get('version', '1.0'),
                    'status': record['status'],
                    'blockchain_tx_id': record.get('blockchain_tx_id'),
                    'action': 'consent' if 'consent_id' in record else 'withdrawal'
                }
                history.append(public_record)
            
            return history
            
        except Exception as e:
            logger.error("Failed to get consent history", user_id=user_id, error=str(e))
            return []
    
    def verify_consent_integrity(self, user_id: str) -> Dict[str, Any]:
        """Verify consent integrity using blockchain"""
        try:
            if user_id not in self.consent_records:
                return {'valid': False, 'reason': 'No consent record found'}
            
            consent = self.consent_records[user_id]
            
            # Verify digital signature
            signature_valid = self._verify_signature(
                consent['consent_hash'], 
                consent['digital_signature']
            )
            
            # Verify blockchain record (simulated)
            blockchain_valid = self._verify_blockchain_record(
                consent.get('blockchain_tx_id'),
                consent['consent_hash']
            )
            
            # Check for tampering
            current_hash = self._create_consent_hash(consent)
            hash_valid = current_hash == consent['consent_hash']
            
            verification_result = {
                'valid': signature_valid and blockchain_valid and hash_valid,
                'signature_valid': signature_valid,
                'blockchain_valid': blockchain_valid,
                'hash_valid': hash_valid,
                'consent_id': consent['consent_id'],
                'verification_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("Consent verification completed", 
                       user_id=user_id, 
                       valid=verification_result['valid'])
            
            return verification_result
            
        except Exception as e:
            logger.error("Consent verification failed", user_id=user_id, error=str(e))
            return {'valid': False, 'error': str(e)}
    
    def _generate_consent_id(self, user_id: str, prefix: str = 'CNS') -> str:
        """Generate unique consent ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:8]
        return f"{prefix}-{timestamp}-{user_hash}"
    
    def _create_consent_hash(self, consent_record: Dict[str, Any]) -> str:
        """Create hash of consent record for integrity"""
        # Remove hash and signature fields for hashing
        record_copy = consent_record.copy()
        record_copy.pop('consent_hash', None)
        record_copy.pop('digital_signature', None)
        record_copy.pop('blockchain_tx_id', None)
        
        # Create deterministic hash
        record_json = json.dumps(record_copy, sort_keys=True)
        return hashlib.sha256(record_json.encode()).hexdigest()
    
    def _sign_consent(self, consent_hash: str) -> str:
        """Create digital signature for consent"""
        try:
            signature = self.private_key.sign(
                consent_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            logger.error("Failed to sign consent", error=str(e))
            raise
    
    def _verify_signature(self, consent_hash: str, signature_hex: str) -> bool:
        """Verify digital signature"""
        try:
            signature = bytes.fromhex(signature_hex)
            self.public_key.verify(
                signature,
                consent_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def _submit_to_blockchain(self, record: Dict[str, Any]) -> str:
        """Submit record to Hyperledger Fabric blockchain"""
        # In production, this would use the actual Hyperledger Fabric SDK
        # For demo, we'll simulate blockchain transaction
        
        try:
            # Simulate blockchain transaction
            tx_id = hashlib.sha256(
                f"{record}{datetime.utcnow()}".encode()
            ).hexdigest()
            
            # In production:
            # client = FabricClient()
            # response = client.chaincode_invoke(
            #     requestor=admin_user,
            #     channel_name=self.channel_name,
            #     peers=['peer0.org1.example.com'],
            #     cc_name=self.chaincode_name,
            #     fcn='recordConsent',
            #     args=[json.dumps(record)]
            # )
            
            logger.info("Blockchain transaction simulated", tx_id=tx_id)
            return tx_id
            
        except Exception as e:
            logger.error("Blockchain submission failed", error=str(e))
            raise
    
    def _verify_blockchain_record(self, tx_id: str, expected_hash: str) -> bool:
        """Verify record exists on blockchain"""
        # In production, this would query the blockchain
        # For demo, we'll simulate verification
        
        if not tx_id:
            return False
        
        # Simulate blockchain query
        # In production:
        # client = FabricClient()
        # response = client.chaincode_query(
        #     requestor=admin_user,
        #     channel_name=self.channel_name,
        #     peers=['peer0.org1.example.com'],
        #     cc_name=self.chaincode_name,
        #     fcn='queryConsent',
        #     args=[tx_id]
        # )
        
        return True  # Simulate successful verification
    
    def get_compliance_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate compliance report for audit purposes"""
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            total_consents = 0
            active_consents = 0
            withdrawn_consents = 0
            expired_consents = 0
            
            consent_types = {
                'general_research': 0,
                'cancer_research_opted_out': 0,
                'mental_health_opted_out': 0,
                'heart_disease_opted_out': 0
            }
            
            for user_id, consent in self.consent_records.items():
                consent_date = datetime.fromisoformat(consent['timestamp'])
                
                if start_dt <= consent_date <= end_dt:
                    total_consents += 1
                    
                    if consent['status'] == 'active':
                        expiry_date = datetime.fromisoformat(consent['expiry_date'])
                        if datetime.utcnow() > expiry_date:
                            expired_consents += 1
                        else:
                            active_consents += 1
                    elif consent['status'] == 'withdrawn':
                        withdrawn_consents += 1
                    
                    # Count consent types
                    if consent.get('general_research_consent'):
                        consent_types['general_research'] += 1
                    if consent.get('cancer_research_opt_out'):
                        consent_types['cancer_research_opted_out'] += 1
                    if consent.get('mental_health_opt_out'):
                        consent_types['mental_health_opted_out'] += 1
                    if consent.get('heart_disease_opt_out'):
                        consent_types['heart_disease_opted_out'] += 1
            
            report = {
                'report_period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'summary': {
                    'total_consents': total_consents,
                    'active_consents': active_consents,
                    'withdrawn_consents': withdrawn_consents,
                    'expired_consents': expired_consents
                },
                'consent_breakdown': consent_types,
                'compliance_metrics': {
                    'consent_rate': (active_consents / total_consents * 100) if total_consents > 0 else 0,
                    'withdrawal_rate': (withdrawn_consents / total_consents * 100) if total_consents > 0 else 0,
                    'expiry_rate': (expired_consents / total_consents * 100) if total_consents > 0 else 0
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
            logger.info("Compliance report generated", 
                       period=f"{start_date} to {end_date}",
                       total_consents=total_consents)
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate compliance report", error=str(e))
            raise