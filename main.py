from src.run_pipeline import run_main_workflow

if __name__ == "__main__":
    print("=== Starting Patch Finder ===")
    run_main_workflow(epochs=2, sandbox_mode=True)
