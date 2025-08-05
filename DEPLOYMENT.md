# Nynava Platform Deployment Guide

## 🚀 Quick Start

### Prerequisites

- **Docker** (v20.10+) and **Docker Compose** (v2.0+)
- **Git** for version control
- **Node.js** (v16+) for local development
- **Python** (v3.8+) for backend development
- **8GB RAM** minimum, 16GB recommended
- **50GB disk space** for blockchain and IPFS data

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/nynava.git
cd nynava

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Configure Environment Variables

Edit `.env` file with your settings:

```bash
# Required Variables
JWT_SECRET_KEY=your-super-secret-jwt-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key
REDIS_PASSWORD=secure-redis-password
GRAFANA_PASSWORD=secure-grafana-password

# Optional Variables
SMTP_USERNAME=your-email@domain.com
SMTP_PASSWORD=your-email-password
SENTRY_DSN=your-sentry-dsn-for-error-tracking
```

### 3. Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f nynava-backend
```

### 4. Initialize the Platform

```bash
# Wait for services to be healthy
docker-compose exec nynava-backend python init_db.py

# Create admin user (optional)
docker-compose exec nynava-backend python create_admin.py

# Test the deployment
curl http://localhost:5000/health
```

## 🌐 Service URLs

After deployment, access these services:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **IPFS Gateway**: http://localhost:8080
- **Grafana Monitoring**: http://localhost:3001 (admin/password from .env)
- **Prometheus Metrics**: http://localhost:9090
- **Flower FL Server**: http://localhost:8080

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Blockchain    │
│   (React/HTML)  │◄──►│ (Python/Flask)  │◄──►│ (Hyperledger)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     IPFS        │    │  Federated      │    │   PostgreSQL    │
│   Storage       │    │   Learning      │    │   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Monitoring & Observability

### Health Checks

```bash
# Check all services
./scripts/health_check.sh

# Individual service checks
curl http://localhost:5000/health
curl http://localhost:5001/api/v0/id  # IPFS
```

### Monitoring Dashboard

1. Open Grafana: http://localhost:3001
2. Login with credentials from `.env`
3. Import Nynava dashboard from `monitoring/grafana/dashboards/`

### Key Metrics to Monitor

- **API Response Times**: < 500ms for uploads
- **IPFS Storage**: Available disk space
- **Blockchain Sync**: Peer connectivity
- **Federated Learning**: Active participants
- **Database**: Connection pool usage

## 🔒 Security Configuration

### SSL/TLS Setup

```bash
# Generate self-signed certificates (development)
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/nynava.key \
  -out nginx/ssl/nynava.crt

# For production, use Let's Encrypt
certbot certonly --webroot -w /var/www/html -d your-domain.com
```

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 22/tcp    # SSH
sudo ufw enable
```

### Database Security

```bash
# Change default passwords
docker-compose exec postgres psql -U nynava_user -d nynava_db
ALTER USER nynava_user WITH PASSWORD 'new-secure-password';
```

## 🚀 Production Deployment

### Cloud Provider Setup

#### AWS Deployment

```bash
# Create EC2 instance (t3.large minimum)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxx

# Setup EBS volumes for persistent data
aws ec2 create-volume --size 100 --volume-type gp3
```

#### Digital Ocean Deployment

```bash
# Create droplet
doctl compute droplet create nynava-prod \
  --size s-4vcpu-8gb \
  --image ubuntu-20-04-x64 \
  --region nyc1 \
  --ssh-keys your-ssh-key-id
```

### Environment-Specific Configurations

#### Production (`docker-compose.prod.yml`)

```yaml
version: '3.8'
services:
  nynava-backend:
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

#### Staging (`docker-compose.staging.yml`)

```yaml
version: '3.8'
services:
  nynava-backend:
    environment:
      - FLASK_ENV=staging
      - FLASK_DEBUG=True
```

### Load Balancing

```nginx
# nginx/nginx.conf
upstream nynava_backend {
    server nynava-backend-1:5000;
    server nynava-backend-2:5000;
    server nynava-backend-3:5000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location /api/ {
        proxy_pass http://nynava_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔄 Backup & Recovery

### Automated Backups

```bash
# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/nynava_$DATE"

# Database backup
docker-compose exec postgres pg_dump -U nynava_user nynava_db > "$BACKUP_DIR/database.sql"

# IPFS data backup
docker-compose exec ipfs tar czf - /data/ipfs > "$BACKUP_DIR/ipfs_data.tar.gz"

# Blockchain data backup
docker cp nynava-hyperledger-peer:/var/hyperledger/production "$BACKUP_DIR/blockchain"
EOF

chmod +x scripts/backup.sh

# Schedule with cron
crontab -e
# Add: 0 2 * * * /path/to/nynava/scripts/backup.sh
```

### Recovery Procedures

```bash
# Restore database
docker-compose exec postgres psql -U nynava_user nynava_db < backup/database.sql

# Restore IPFS data
docker-compose down
docker volume rm nynava_ipfs-data
docker volume create nynava_ipfs-data
docker run --rm -v nynava_ipfs-data:/data -v $(pwd)/backup:/backup alpine tar xzf /backup/ipfs_data.tar.gz -C /data
docker-compose up -d
```

## 🧪 Testing & Validation

### Automated Testing

```bash
# Run backend tests
docker-compose exec nynava-backend python -m pytest tests/

# Run frontend tests
cd frontend && npm test

# Integration tests
./scripts/run_integration_tests.sh
```

### Privacy Compliance Testing

```bash
# Test data anonymization
python tests/test_anonymization.py

# HIPAA compliance check
python tests/test_hipaa_compliance.py

# Blockchain consent verification
python tests/test_consent_blockchain.py
```

### Performance Testing

```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:5000/api/v1/health

# Upload performance test
python tests/performance/test_upload_performance.py
```

## 🐛 Troubleshooting

### Common Issues

#### Backend Won't Start

```bash
# Check logs
docker-compose logs nynava-backend

# Common fixes
docker-compose down
docker-compose pull
docker-compose up -d --force-recreate
```

#### IPFS Connection Issues

```bash
# Reset IPFS node
docker-compose stop ipfs
docker volume rm nynava_ipfs-data
docker-compose up -d ipfs
```

#### Database Connection Errors

```bash
# Check database status
docker-compose exec postgres pg_isready -U nynava_user

# Reset database
docker-compose down
docker volume rm nynava_postgres-data
docker-compose up -d postgres
```

### Log Analysis

```bash
# Centralized logging
docker-compose logs -f --tail=100

# Specific service logs
docker-compose logs nynava-backend
docker-compose logs ipfs
docker-compose logs hyperledger-peer
```

## 📈 Scaling & Optimization

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  nynava-backend:
    deploy:
      replicas: 5
  
  postgres:
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == manager
```

### Performance Optimization

```bash
# Database optimization
docker-compose exec postgres psql -U nynava_user -d nynava_db -c "
  CREATE INDEX CONCURRENTLY idx_user_data_created_at ON user_data(created_at);
  CREATE INDEX CONCURRENTLY idx_consent_records_user_id ON consent_records(user_id);
"

# Redis memory optimization
docker-compose exec redis redis-cli CONFIG SET maxmemory 2gb
docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## 🔐 Security Hardening

### Container Security

```bash
# Scan for vulnerabilities
docker-compose exec nynava-backend pip-audit

# Update base images
docker-compose pull
docker-compose up -d --force-recreate
```

### Network Security

```bash
# Restrict network access
iptables -A INPUT -p tcp --dport 5432 -s 172.20.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -j DROP
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks

```bash
# Weekly maintenance script
cat > scripts/weekly_maintenance.sh << 'EOF'
#!/bin/bash
# Update containers
docker-compose pull
docker-compose up -d --force-recreate

# Clean up old data
docker system prune -f
docker volume prune -f

# Database maintenance
docker-compose exec postgres vacuumdb -U nynava_user -d nynava_db -z

# Backup verification
./scripts/verify_backups.sh
EOF
```

### Getting Help

- **Documentation**: [docs.nynava.com](https://docs.nynava.com)
- **GitHub Issues**: [github.com/your-org/nynava/issues](https://github.com/your-org/nynava/issues)
- **Community Discord**: [discord.gg/nynava](https://discord.gg/nynava)
- **Email Support**: support@nynava.com

## 📋 Deployment Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database initialized
- [ ] IPFS node synchronized
- [ ] Blockchain network connected
- [ ] Monitoring dashboards configured
- [ ] Backup procedures tested
- [ ] Security hardening applied
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Team trained on operations

---

**🎉 Congratulations!** Your Nynava platform is now deployed and ready to revolutionize healthcare AI while maintaining the highest standards of privacy and security.