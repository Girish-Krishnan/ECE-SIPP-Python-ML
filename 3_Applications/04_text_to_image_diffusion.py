from diffusers import StableDiffusionPipeline
import torch

def make_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")

    # fp16 only on CUDA (MPS fp16 still less stable in some versions)
    dtype = torch.float16 if has_cuda else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        # keep safety checker unless you are sure you want it off
        # safety_checker=None
    )

    pipe.to(device)

    # Memory friendly tweaks that are safe on all devices
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except AttributeError:
        pass

    # CUDA only optimizations
    if has_cuda:
        # Sequential / model cpu offload only makes sense with CUDA
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass
        # autocast context will be added in main
    return pipe, device, has_cuda

def main():
    pipe, device, has_cuda = make_pipeline()
    prompt = input("Describe the image you want to create: ")

    if has_cuda:
        autocast_ctx = torch.autocast("cuda")
    else:
        # Do not use autocast for mps or cpu here
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    with autocast_ctx:
        image = pipe(prompt, num_inference_steps=30).images[0]

    image.save("generated.png")
    print("Saved generated.png")

if __name__ == "__main__":
    main()
