import time
import pickle

import numpy as np
import zmq

class ModelClient:
    """Client for model inference server."""
    
    def __init__(self, host="localhost", port=5555, timeout_ms: int = 30000, retries: int = 0):
        """
        Initialize client connection.
        
        Args:
            host: Server hostname or IP
            port: Server port
        """
        self.host = host
        self.port = port
        self.timeout_ms = int(timeout_ms)
        self.retries = int(retries)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)  # Set receive timeout
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)  # Set send timeout
        self.socket.setsockopt(zmq.LINGER, 0)
        
        print(f"Connected to model server at {host}:{port}")

    def _request(self, request: dict) -> dict:
        """Send a request and receive a response with basic diagnostics.

        If you see a timeout here, it's almost always one of:
        - The server/bridge isn't running on host:port.
        - The server/bridge crashed before replying.
        - Modal cold start (model download/load) is taking longer than timeout_ms.
        """
        payload = pickle.dumps(request)
        attempt = 0
        while True:
            attempt += 1
            try:
                self.socket.send(payload)
                response_bytes = self.socket.recv()
                response = pickle.loads(response_bytes)
                return response
            except zmq.error.Again as e:
                if attempt > (1 + self.retries):
                    raise TimeoutError(
                        "Timed out waiting for a response from the model server "
                        f"at {self.host}:{self.port} after {self.timeout_ms}ms. "
                        "Ensure the server is running and reachable. "
                        "For local inference, run scripts/serve.py. "
                        "For Modal inference, run scripts/serve_modal.py (and ensure modal_app.py is deployed). "
                        "If this is a Modal cold start, increase timeout_ms or retries."
                    ) from e
                time.sleep(0.25)
            except Exception:
                # ZMQ REQ sockets are strict (send/recv alternation). If something failed mid-flight,
                # rebuild the socket so the next request doesn't get stuck.
                try:
                    self.socket.close(0)
                except Exception:
                    pass
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect(f"tcp://{self.host}:{self.port}")
                self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                self.socket.setsockopt(zmq.LINGER, 0)
                raise
    
    def predict(self, image: np.ndarray) -> dict:
        """
        Send an image and receive predicted actions.
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            List of action dicts, each containing:
                - j_left: [x, y] left joystick position
                - j_right: [x, y] right joystick position  
                - buttons: list of button values
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        request = {
            "type": "predict",
            "image": image
        }

        response = self._request(request)
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["pred"]
    
    def reset(self):
        """Reset the server's session (clear buffers)."""
        request = {"type": "reset"}

        response = self._request(request)
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        print("Session reset")

    def info(self) -> dict:
        """Get session info from the server."""
        request = {"type": "info"}

        response = self._request(request)
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["info"]

    def close(self):
        """Close the connection."""
        self.socket.close()
        self.context.term()
        print("Connection closed")
    
    def __enter__(self):
        """Support for context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context."""
        self.close()
