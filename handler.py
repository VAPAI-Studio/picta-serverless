import runpod
from comfyui_handler import handle_comfyui_workflow
from recolor_handler import handle_recolor_background


def handler(job):
    """
    Main handler that routes between ComfyUI workflows and background recoloring
    based on the task type.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with results.
    """

    job_input = job["input"]
    job_id = job["id"]

    # Check if this is a background recoloring task
    if isinstance(job_input, dict) and job_input.get("task_type") == "recolor_background":
        return handle_recolor_background(job_input, job_id)
    
    # Otherwise, handle as ComfyUI workflow
    return handle_comfyui_workflow(job_input, job_id)


if __name__ == "__main__":
    print("worker-comfyui - Starting handler...")
    runpod.serverless.start({"handler": handler})