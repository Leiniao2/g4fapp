GPT4Free Web Application
A Flask-based web application that provides a user-friendly interface for accessing various AI models through the GPT4Free library. The application can be deployed on Google App Engine or any other cloud platform.

Features
Web Interface: Clean, responsive web interface for chatting with AI models
Multiple Providers: Support for various AI providers through GPT4Free
Model Selection: Choose from different AI models and providers
Streaming Support: Real-time streaming responses for better user experience
Conversation History: Maintains chat history during the session
Error Handling: Robust error handling and user feedback
Cloud Ready: Configured for easy deployment on Google App Engine
File Structure
gpt4free-app/
├── main.py              # Main Flask application
├── templates/
│   └── index.html       # Web interface template
├── requirements.txt     # Python dependencies
├── app.yaml            # Google App Engine configuration
├── Dockerfile          # Docker configuration (alternative)
└── README.md           # This file
Installation
Local Development
Clone or create the project structure with the provided files
Install dependencies:
bash
pip install -r requirements.txt
Run the application:
bash
python main.py
Open your browser and navigate to http://localhost:8080
Google App Engine Deployment
Install Google Cloud SDK and authenticate:
bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
Deploy the application:
bash
gcloud app deploy
Open the deployed app:
bash
gcloud app browse
Docker Deployment (Alternative)
Build the Docker image:
bash
docker build -t gpt4free-app .
Run the container:
bash
docker run -p 8080:8080 gpt4free-app
Usage
Access the web interface at your deployed URL or http://localhost:8080
Select a provider from the dropdown (or leave as "Auto" for best available)
Choose a model (or leave as "Default Model")
Toggle streaming if you want real-time responses
Type your message and click "Send" or press Enter
View the AI response in the chat interface
API Endpoints
GET / - Main web interface
GET /api/providers - Get available providers
GET /api/models - Get available models
POST /api/generate - Generate AI responses
GET /api/health - Health check endpoint
Configuration
Environment Variables
SECRET_KEY - Flask secret key for sessions
PORT - Port to run the application (default: 8080)
Google App Engine Settings
The app.yaml file includes:

Runtime: Python 3.9
Automatic scaling configuration
Static file handling
Security settings
Key Features Explained
Provider Selection
The application automatically detects available providers from the GPT4Free library and presents them in a dropdown. Each provider has different capabilities and availability.

Model Selection
Various AI models are available through different providers. The application lists all available models and their information.

Streaming Responses
When enabled, responses are streamed in real-time, providing a better user experience for longer responses.

Error Handling
The application includes comprehensive error handling for:

Network timeouts
Provider failures
Invalid requests
Server errors
Security Considerations
Change the SECRET_KEY in production
Consider implementing rate limiting
Add authentication if needed
Monitor usage to prevent abuse
Dependencies
Flask: Web framework
g4f: GPT4Free library for AI providers
gunicorn: WSGI HTTP server
aiohttp: Async HTTP client
asyncio-throttle: Rate limiting
nest-asyncio: Nested async support
Troubleshooting
Common Issues
Provider not working: Some providers may be temporarily unavailable. Try switching to a different provider.
Timeout errors: Increase the timeout in app.yaml or try a different provider.
Memory issues: Adjust the instance class in app.yaml if needed.
Rate limiting: Some providers may have rate limits. Implement appropriate delays.
Debugging
Enable debug mode in development:

python
app.run(debug=True)
Check logs in Google App Engine:

bash
gcloud app logs tail -s default
Limitations
Some providers may have usage limits
Response quality varies by provider
Not all models support all features
Streaming may not work with all providers
Contributing
Fork the repository
Create a feature branch
Make your changes
Test thoroughly
Submit a pull request
License
This project is open source. Please check the GPT4Free library license for any restrictions.

Support
For issues related to:

This application: Create an issue in the repository
GPT4Free library: Check the official GPT4Free documentation
Google App Engine: Consult Google Cloud documentation
Disclaimer
This application is for educational and research purposes. Be mindful of the terms of service of the underlying AI providers and use responsibly.

