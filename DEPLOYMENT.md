# Pollen ID - Deployment Guide
# ==============================

## 🚀 Deploying to Streamlit Cloud

### Step 1: Prepare Your GitHub Repository

1. **Create a new GitHub repository**
   ```bash
   # Initialize git (if not already done)
   git init
   
   # Add files
   git add .
   
   # Commit
   git commit -m "Initial commit: Pollen ID MVP"
   
   # Add remote
   git remote add origin https://github.com/YOUR_USERNAME/pollen-id.git
   
   # Push
   git push -u origin main
   ```

2. **Verify your repository structure**
   ```
   your-repo/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   ├── .gitignore
   └── .env.example
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Sign in to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign in with GitHub"
   - Authorize Streamlit to access your repositories

2. **Create new app**
   - Click "New app" button
   - Repository: Select your `pollen-id` repository
   - Branch: `main` (or your default branch)
   - Main file path: `app.py`
   - App URL: Choose a custom URL or use auto-generated

3. **Configure secrets**
   - Click "Advanced settings" (before deploying)
   - Or go to app settings → Secrets after deployment
   - Add the following in TOML format:

   ```toml
   # .streamlit/secrets.toml format
   GOOGLE_API_KEY = "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
   OPENAI_API_KEY = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
   ```

4. **Deploy**
   - Click "Deploy!"
   - Wait 2-5 minutes for initial deployment
   - Your app will be live at: `https://your-app-name.streamlit.app`

### Step 3: Verify Deployment

1. **Check app status**
   - Green dot = Running
   - Yellow dot = Building
   - Red dot = Error (check logs)

2. **Test functionality**
   - [ ] App loads without errors
   - [ ] API keys are accessible (test with a location)
   - [ ] Charts render correctly
   - [ ] Health chat responds

3. **Monitor logs**
   - Click "Manage app" → "Logs"
   - Look for any error messages
   - Common issues:
     - Missing dependencies → Update `requirements.txt`
     - API key errors → Check secrets configuration
     - Import errors → Verify Python version compatibility

### Step 4: Custom Domain (Optional)

1. **Streamlit Cloud subdomain**
   - Free tier: `your-app.streamlit.app`
   - Cannot customize subdomain on free tier

2. **Custom domain (requires upgrade)**
   - Go to app settings → General
   - Add custom domain
   - Update DNS records (CNAME)
   - Enable HTTPS (automatic)

---

## 🐳 Alternative: Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Commands

```bash
# Build image
docker build -t pollen-id:latest .

# Run container
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  pollen-id:latest

# Or use docker-compose
docker-compose up -d
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  pollen-id:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## ☁️ Alternative: AWS/GCP/Azure

### AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize**
   ```bash
   eb init -p python-3.11 pollen-id
   ```

3. **Create environment**
   ```bash
   eb create pollen-id-env
   ```

4. **Set environment variables**
   ```bash
   eb setenv GOOGLE_API_KEY=xxx OPENAI_API_KEY=xxx
   ```

5. **Deploy**
   ```bash
   eb deploy
   ```

### Google Cloud Run

1. **Create Dockerfile** (see above)

2. **Build and push**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/pollen-id
   ```

3. **Deploy**
   ```bash
   gcloud run deploy pollen-id \
     --image gcr.io/PROJECT_ID/pollen-id \
     --platform managed \
     --set-env-vars GOOGLE_API_KEY=xxx,OPENAI_API_KEY=xxx
   ```

### Azure Web Apps

1. **Create App Service**
   ```bash
   az webapp create \
     --resource-group myResourceGroup \
     --plan myAppServicePlan \
     --name pollen-id \
     --runtime "PYTHON:3.11"
   ```

2. **Deploy**
   ```bash
   az webapp up --name pollen-id
   ```

3. **Set config**
   ```bash
   az webapp config appsettings set \
     --name pollen-id \
     --settings GOOGLE_API_KEY=xxx OPENAI_API_KEY=xxx
   ```

---

## 🔐 Security Best Practices

### API Key Management

1. **Never commit API keys**
   - Use `.gitignore` to exclude `.env` files
   - Use secrets management (Streamlit secrets, AWS Secrets Manager, etc.)

2. **Restrict API keys**
   - Google Cloud: Add HTTP referrer restrictions
   - OpenAI: Set usage limits and monitoring

3. **Rotate keys regularly**
   - Monthly rotation recommended
   - Use key versioning

### App Security

1. **Rate limiting**
   ```python
   # Add to app.py
   from streamlit_extras.throttle import throttle
   
   @throttle(seconds=5)
   def fetch_pollen_data():
       # ... existing code
   ```

2. **Input validation**
   - Already implemented in `get_coordinates()`
   - Sanitize user inputs

3. **HTTPS only**
   - Streamlit Cloud: Automatic HTTPS
   - Custom deployments: Use reverse proxy (nginx)

---

## 📊 Monitoring & Analytics

### Streamlit Cloud Analytics

1. **Built-in metrics**
   - App settings → Analytics
   - View user count, session duration

2. **Custom logging**
   ```python
   import logging
   
   logging.info(f"User searched location: {location}")
   ```

### External Monitoring

1. **Google Analytics**
   ```python
   # Add to app.py
   import streamlit.components.v1 as components
   
   components.html("""
   <!-- Google Analytics tag -->
   """)
   ```

2. **Sentry (Error Tracking)**
   ```python
   pip install sentry-sdk
   
   import sentry_sdk
   sentry_sdk.init(dsn="your-dsn")
   ```

---

## 🔧 Troubleshooting Deployment

### Common Deployment Errors

**Error: "Requirements installation failed"**
- Solution: Pin dependency versions in `requirements.txt`
- Check Python version compatibility (use Python 3.9-3.11)

**Error: "ModuleNotFoundError"**
- Solution: Ensure all imports are in `requirements.txt`
- Try: `pip freeze > requirements.txt`

**Error: "App keeps restarting"**
- Solution: Check logs for infinite loops or crashes
- Reduce API call frequency

**Error: "Secrets not found"**
- Solution: Verify secrets.toml format (TOML, not JSON)
- Check for typos in secret keys

### Performance Issues

**Slow loading**
- Add `@st.cache_data` to expensive functions
- Reduce chart complexity
- Optimize API calls

**Memory errors**
- Reduce data retention
- Clear cache periodically
- Consider upgrading Streamlit Cloud tier

---

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: [Your repo]/issues

---

**Deployment Checklist**
- [ ] Code pushed to GitHub
- [ ] `requirements.txt` updated
- [ ] API keys configured in secrets
- [ ] App deployed successfully
- [ ] Functionality tested
- [ ] Error monitoring enabled
- [ ] Custom domain configured (optional)
