from flask import Flask, render_template, request, jsonify, Response
import g4f
import json
import logging
from typing import Dict, List, Any
import os
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4FreeService:
    def __init__(self):
        self.providers = self._get_available_providers()
        self.models = self._get_available_models()
    
    def _get_available_providers(self) -> Dict[str, Any]:
        """Get available providers from g4f"""
        providers = {}
        try:
            # Common working providers
            provider_list = [
                'Bing', 'ChatGpt', 'GPTalk', 'Liaobots', 'Phind',
                'Yqcloud', 'You', 'Aichat', 'ChatBase', 'OpenaiChat'
            ]
            
            for provider_name in provider_list:
                try:
                    if hasattr(g4f.Provider, provider_name):
                        provider = getattr(g4f.Provider, provider_name)
                        providers[provider_name] = {
                            'name': provider_name,
                            'working': True,
                            'supports_stream': True,
                            'supports_system_message': True,
                            'url': getattr(provider, 'url', '')
                        }
                except Exception as e:
                    logger.warning(f"Provider {provider_name} not available: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting providers: {e}")
        return providers
    
    def _get_available_models(self) -> Dict[str, Any]:
        """Get available models from g4f"""
        models = {}
        try:
            # Common working models
            model_list = [
                'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'claude-v1',
                'claude-instant-v1', 'palm', 'llama-2-7b', 'llama-2-13b'
            ]
            
            for model_name in model_list:
                models[model_name] = {
                    'name': model_name,
                    'base_provider': '',
                    'best_provider': ''
                }
                
        except Exception as e:
            logger.error(f"Error getting models: {e}")
        return models
    
    def generate_response(self, messages: List[Dict], provider_name: str = None, model_name: str = None, stream: bool = False):
        """Generate response using g4f"""
        try:
            # Get provider
            provider = None
            if provider_name and hasattr(g4f.Provider, provider_name):
                provider = getattr(g4f.Provider, provider_name)
            
            # Get model - use string format for g4f
            model = model_name if model_name else 'gpt-3.5-turbo'
            
            # Generate response
            if stream:
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    provider=provider,
                    stream=True
                )
                return response
            else:
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    provider=provider,
                    stream=False
                )
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise e

# Initialize service
gpt4free_service = GPT4FreeService()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         providers=gpt4free_service.providers,
                         models=gpt4free_service.models)

@app.route('/api/providers')
def get_providers():
    """Get available providers"""
    return jsonify(gpt4free_service.providers)

@app.route('/api/models')
def get_models():
    """Get available models"""
    return jsonify(gpt4free_service.models)

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate response from GPT4Free"""
    try:
        data = request.json
        messages = data.get('messages', [])
        provider_name = data.get('provider')
        model_name = data.get('model')
        stream = data.get('stream', False)
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Simple validation and formatting
        formatted_messages = []
        for msg in messages:
            if 'role' in msg and 'content' in msg:
                formatted_messages.append({
                    'role': msg['role'],
                    'content': str(msg['content'])
                })
        
        if not formatted_messages:
            return jsonify({'error': 'Invalid message format'}), 400
        
        try:
            if stream:
                def generate_stream():
                    try:
                        response = gpt4free_service.generate_response(
                            formatted_messages, provider_name, model_name, stream=True
                        )
                        
                        # Handle streaming response
                        for chunk in response:
                            if chunk:
                                yield f"data: {json.dumps({'content': chunk})}\n\n"
                        
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                return Response(
                    generate_stream(),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no'
                    }
                )
            else:
                response = gpt4free_service.generate_response(
                    formatted_messages, provider_name, model_name, stream=False
                )
                return jsonify({'response': response})
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'providers_count': len(gpt4free_service.providers),
        'models_count': len(gpt4free_service.models)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
