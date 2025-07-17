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
        self.working_providers = self._test_providers()
    
    def _get_available_providers(self) -> Dict[str, Any]:
        """Get available providers from g4f"""
        providers = {}
        try:
            # Most reliable providers (updated list)
            provider_list = [
                'You', 'Aichat', 'ChatBase', 'FreeGpt', 'GPTalk', 
                'Liaobots', 'Phind', 'Yqcloud', 'Bing'
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
        models = {
            'gpt-3.5-turbo': {'name': 'GPT-3.5 Turbo', 'base_provider': '', 'best_provider': ''},
            'gpt-4': {'name': 'GPT-4', 'base_provider': '', 'best_provider': ''},
            'claude-v1': {'name': 'Claude v1', 'base_provider': '', 'best_provider': ''},
            'palm': {'name': 'PaLM', 'base_provider': '', 'best_provider': ''},
        }
        return models
    
    def _test_providers(self) -> List[str]:
        """Test providers to see which ones are working"""
        working = []
        test_messages = [{"role": "user", "content": "Hi"}]
        
        for provider_name in self.providers.keys():
            try:
                provider = getattr(g4f.Provider, provider_name)
                # Quick test
                response = g4f.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=test_messages,
                    provider=provider,
                    stream=False,
                    timeout=10
                )
                if response and len(str(response).strip()) > 0:
                    working.append(provider_name)
                    logger.info(f"Provider {provider_name} is working")
                else:
                    logger.warning(f"Provider {provider_name} returned empty response")
            except Exception as e:
                logger.warning(f"Provider {provider_name} test failed: {e}")
                continue
        
        logger.info(f"Working providers: {working}")
        return working
    
    def generate_response(self, messages: List[Dict], provider_name: str = None, model_name: str = None, stream: bool = False):
        """Generate response using g4f with fallback mechanism"""
        
        # Clean and validate messages
        clean_messages = []
        for msg in messages:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                clean_messages.append({
                    'role': str(msg['role']).lower(),
                    'content': str(msg['content']).strip()
                })
        
        if not clean_messages:
            raise ValueError("No valid messages provided")
        
        # Get model
        model = model_name if model_name else 'gpt-3.5-turbo'
        
        # Try specified provider first, then fallback to working providers
        providers_to_try = []
        if provider_name and provider_name in self.providers:
            providers_to_try.append(provider_name)
        
        # Add working providers as fallback
        providers_to_try.extend([p for p in self.working_providers if p not in providers_to_try])
        
        # If no working providers, try all available providers
        if not providers_to_try:
            providers_to_try = list(self.providers.keys())
        
        # Try providers one by one
        last_error = None
        for provider_name in providers_to_try:
            try:
                logger.info(f"Trying provider: {provider_name}")
                provider = getattr(g4f.Provider, provider_name)
                
                if stream:
                    response = g4f.ChatCompletion.create(
                        model=model,
                        messages=clean_messages,
                        provider=provider,
                        stream=True,
                        timeout=30
                    )
                else:
                    response = g4f.ChatCompletion.create(
                        model=model,
                        messages=clean_messages,
                        provider=provider,
                        stream=False,
                        timeout=30
                    )
                
                # Validate response
                if response:
                    if stream:
                        return response  # Return generator for streaming
                    else:
                        response_str = str(response).strip()
                        if response_str and len(response_str) > 0:
                            logger.info(f"Success with provider: {provider_name}")
                            return response_str
                        else:
                            logger.warning(f"Empty response from provider: {provider_name}")
                            continue
                else:
                    logger.warning(f"No response from provider: {provider_name}")
                    continue
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        # If all providers failed, try without specifying provider
        try:
            logger.info("Trying without specific provider")
            if stream:
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=clean_messages,
                    stream=True,
                    timeout=30
                )
            else:
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=clean_messages,
                    stream=False,
                    timeout=30
                )
            
            if response:
                if stream:
                    return response
                else:
                    response_str = str(response).strip()
                    if response_str:
                        return response_str
                        
        except Exception as e:
            last_error = e
            logger.error(f"Final attempt failed: {e}")
        
        # If everything failed
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def refresh_working_providers(self):
        """Refresh the list of working providers"""
        self.working_providers = self._test_providers()
        return self.working_providers

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

@app.route('/api/refresh-providers', methods=['POST'])
def refresh_providers():
    """Refresh working providers"""
    try:
        working = gpt4free_service.refresh_working_providers()
        return jsonify({
            'working_providers': working,
            'total_providers': len(gpt4free_service.providers)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        logger.info(f"Generate request: provider={provider_name}, model={model_name}, stream={stream}")
        logger.info(f"Messages: {messages}")
        
        try:
            if stream:
                def generate_stream():
                    try:
                        response = gpt4free_service.generate_response(
                            messages, provider_name, model_name, stream=True
                        )
                        
                        # Handle streaming response
                        content_sent = False
                        for chunk in response:
                            if chunk and str(chunk).strip():
                                content_sent = True
                                yield f"data: {json.dumps({'content': str(chunk)})}\n\n"
                        
                        if not content_sent:
                            yield f"data: {json.dumps({'error': 'No content received from provider'})}\n\n"
                        else:
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
                    messages, provider_name, model_name, stream=False
                )
                
                if response and str(response).strip():
                    return jsonify({'response': str(response)})
                else:
                    return jsonify({'error': 'No response generated'}), 500
                
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
        'models_count': len(gpt4free_service.models),
        'working_providers': gpt4free_service.working_providers,
        'working_count': len(gpt4free_service.working_providers)
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
