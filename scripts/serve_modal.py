import zmq
import argparse
import pickle
import modal
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Modal Inference Bridge")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--app-name", type=str, default="nitrogen-inference", help="Name of the Modal app")
    args = parser.parse_args()

    print(f"Looking up Modal app '{args.app_name}'...")
    try:
        # Connect to the deployed Modal class
        Model = modal.Cls.from_name(args.app_name, "Model")
        model = Model()
        # Verify connection/warmup by calling info (optional but good for debugging)
        print("Warming up / Verifying connection...")
        print("Note: If this is the first run, the model might be downloading or loading.")
        print("You can view the server logs in another terminal with: modal app logs nitrogen-inference")
        info = model.info.remote()
        print(f"Connected to Modal. Session Info: {info}")
    except Exception as e:
        print(f"Failed to connect to Modal app '{args.app_name}': {e}")
        print("Please ensure you have deployed the app:")
        print("  modal deploy modal_app.py")
        sys.exit(1)

    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    # Create poller
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"\n{'='*60}")
    print(f"Bridge Server running on port {args.port}")
    print(f"Forwarding requests to Modal app '{args.app_name}'")
    print(f"Waiting for requests...")
    print(f"{ '='*60}\n")

    try:
        while True:
            # Poll with 100ms timeout to allow interrupt handling
            events = dict(poller.poll(timeout=100))
            if socket in events and events[socket] == zmq.POLLIN:
                request_bytes = socket.recv()
                try:
                    request = pickle.loads(request_bytes)
                except Exception as e:
                    print(f"Error decoding request: {e}")
                    socket.send(pickle.dumps({"status": "error", "message": "Pickle decode error"}))
                    continue

                req_type = request.get("type")
                
                if req_type == "reset":
                    print("Forwarding RESET...")
                    model.reset.remote()
                    response = {"status": "ok"}
                    print("Reset complete.")
                
                elif req_type == "info":
                    # print("Forwarding INFO...")
                    info = model.info.remote()
                    response = {"status": "ok", "info": info}
                
                elif req_type == "predict":
                    # Optional: Print a dot for activity or suppress for speed
                    # print(".", end="", flush=True) 
                    image = request["image"]
                    try:
                        result = model.predict.remote(image)
                        response = {
                            "status": "ok",
                            "pred": result
                        }
                    except Exception as e:
                        print(f"\nPrediction Error: {e}")
                        response = {"status": "error", "message": str(e)}
                
                else:
                    response = {"status": "error", "message": f"Unknown request type: {req_type}"}

                # Send response
                socket.send(pickle.dumps(response))
                
    except KeyboardInterrupt:
        print("\nShutting down bridge...")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
