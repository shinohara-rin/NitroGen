import zmq
import argparse
import pickle
import modal
import sys
import numpy as np
import os
import time
import traceback

def main():
    parser = argparse.ArgumentParser(description="Modal Inference Bridge")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--app-name", type=str, default="nitrogen-inference", help="Name of the Modal app")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    verbose = args.verbose or os.environ.get("NITROGEN_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}

    print(f"Looking up Modal app '{args.app_name}'...")
    try:
        # Connect to the deployed Modal class
        Model = modal.Cls.from_name(args.app_name, "Model")
        model = Model()
        # Verify connection/warmup by calling info (optional but good for debugging)
        print("Warming up / Verifying connection...")
        print("Note: If this is the first run, the model might be downloading or loading.")
        print("You can view the server logs in another terminal with: modal app logs nitrogen-inference")
        t0 = time.time()
        info = model.info.remote()
        dt = time.time() - t0
        print(f"Connected to Modal. Session Info: {info}")
        print(f"Modal warmup/info call latency: {dt:.3f}s")
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
                if verbose:
                    print(f"[bridge] received {len(request_bytes)} bytes")
                try:
                    request = pickle.loads(request_bytes)
                except Exception as e:
                    print(f"Error decoding request: {e}")
                    if verbose:
                        print(traceback.format_exc())
                    socket.send(pickle.dumps({"status": "error", "message": "Pickle decode error"}))
                    continue

                req_type = request.get("type")
                if verbose:
                    print(f"[bridge] request type={req_type!r}")
                
                if req_type == "reset":
                    print("Forwarding RESET...")
                    t0 = time.time()
                    try:
                        model.reset.remote()
                        response = {"status": "ok"}
                        print(f"Reset complete. latency={time.time() - t0:.3f}s")
                    except Exception as e:
                        print(f"\nReset Error: {e}")
                        if verbose:
                            print(traceback.format_exc())
                        response = {"status": "error", "message": str(e)}
                
                elif req_type == "info":
                    # print("Forwarding INFO...")
                    t0 = time.time()
                    try:
                        info = model.info.remote()
                        response = {"status": "ok", "info": info}
                    except Exception as e:
                        print(f"\nInfo Error: {e}")
                        if verbose:
                            print(traceback.format_exc())
                        response = {"status": "error", "message": str(e)}
                    if verbose:
                        print(f"[bridge] info latency={time.time() - t0:.3f}s")
                
                elif req_type == "predict":
                    # Optional: Print a dot for activity or suppress for speed
                    # print(".", end="", flush=True) 
                    image = request["image"]
                    if verbose:
                        if isinstance(image, np.ndarray):
                            print(
                                f"[bridge] image ndarray shape={image.shape} dtype={image.dtype} "
                                f"nbytes={image.nbytes}"
                            )
                        else:
                            print(f"[bridge] image type={type(image)}")
                    try:
                        t0 = time.time()
                        result = model.predict.remote(image)
                        dt = time.time() - t0
                        response = {
                            "status": "ok",
                            "pred": _to_builtin(result)
                        }
                        if verbose:
                            keys = list(result.keys()) if isinstance(result, dict) else None
                            print(f"[bridge] predict latency={dt:.3f}s result_keys={keys}")
                    except Exception as e:
                        print(f"\nPrediction Error: {e}")
                        if verbose:
                            print(traceback.format_exc())
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


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _to_builtin(tolist())
    return obj
