from flask import Flask, request, Response, stream_with_context, json
from flask_cors import CORS 
from agent import run_Agent
import sys

app = Flask(__name__)
CORS(app) 

@app.route("/agent", methods=["POST"])
def agent_endpoint():
    """
    Flask endpoint that streams the response from the agent.
    
    The frontend expects a streaming response body.
    """
    try:
        data = request.json
        
        if not data or 'input' not in data:
            return Response("Missing 'input' field in JSON body.", status=400)
            
        input_text = data.get("input")
        
        def generate():
            try:
                for chunk in run_Agent(input_text):
                    yield chunk
            except Exception as e:
                error_message = f"\n\n[Agent Error: Could not process request. {str(e)}]"
                print(f"Agent streaming error: {e}", file=sys.stderr)
                yield error_message
        return Response(stream_with_context(generate()), mimetype='text/plain')

    except Exception as e:
        print(f"API endpoint error: {e}", file=sys.stderr)
        return Response(f"Internal Server Error: Could not initialize request: {str(e)}", status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)