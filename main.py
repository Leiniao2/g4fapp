from flask import Flask, render_template, request, jsonify, stream_template
import g4f
import asyncio
import json
import logging
from typing import Dict, List, Any
import os
from concurrent.futures import ThreadPoolExecutor
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=10)

class GPT4FreeService:
    def __init__(self):
        self.providers = self._get_available_providers()
        self.models = self._get_available_models()
    
    def _get_available_providers(self) -> Dict[str, Any]:
        """Get available providers from g4f"""
        providers = {}
        try:
            for provider_name in dir(g4f.Provider):
                if not provider_name.startswith('_'):
                    provider = getattr(g4f.Provider, provider_name)
                    if hasattr(provider, 'working') and provider.working:
                        providers[provider_name] = {
                            'name': provider_name,
                            'working': provider.working,
                            'supports_stream': getattr(provider, 'supports_stream', False),
                            'supports_system_message': getattr(provider, 'supports_system_message', True),
                            'url': getattr(provider, 'url', '')
                        }
        except Exception as e:
            logger.error(f"Error getting providers: {e}")
        return providers
    
    def _get_available_models(self) -> Dict[str, Any]:
        """Get available models from g4f"""
        models = {}
        try:
            for model_name in dir(g4f.models):
                if not model_name.startswith('_'):
                    model = getattr(g4f.models, model_name)
                    if hasattr(model, 'name'):
                        models[model_name] = {
                            'name': model.name,
                            'base_provider': getattr(model, 'base_provider', ''),
                            'best_provider': getattr(model, 'best_provider', '')
                        }
        except Exception as e:
            logger.error(f"Error getting models: {e}")
        return models
    
    async def generate_response(self, messages: List[Dict], provider_name: str = None, model_name: str = None, stream: bool = False):
        """Generate response using g4f"""
        try:
            # Get provider
            provider = None
            if provider_name:
                provider = getattr(g4f.Provider, provider_name, None)
            
            # Get model
            model = g4f.models.default
            if model_name:
                model = getattr(g4f.models, model_name, g4f.models.default)
            
            # Generate response
            if stream:
                response = await g4f.ChatCompletion.create_async(
                    model=model,
                    messages=messages,
                    provider=provider,
                    stream=True
                )
                return response
            else:
                response = await g4f.ChatCompletion.create_async(
                    model=model,
                    messages=messages,
                    provider=provider
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
        
        # Run async function in thread pool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if stream:
                def generate_stream():
                    try:
                        response = loop.run_until_complete(
                            gpt4free_service.generate_response(
                                messages, provider_name, model_name, stream=True
                            )
                        )
                        for chunk in response:
                            if chunk:
                                yield f"data: {json.dumps({'content': chunk})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    finally:
                        loop.close()
                
                return app.response_class(
                    generate_stream(),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no'
                    }
                )
            else:
                response = loop.run_until_complete(
                    gpt4free_service.generate_response(
                        messages, provider_name, model_name, stream=False
                    )
                )
                return jsonify({'response': response})
        finally:
            loop.close()
            
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
