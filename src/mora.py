import time
import torch
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image


def load_models():
    # Load the stable video diffusion pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    # Load the stable diffusion pipeline for base and refiner
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    # Define how many steps and what % of steps to be run on each expert (80/20)
    n_steps = 40
    high_noise_frac = 0.8

    return pipe, base, refiner, n_steps, high_noise_frac


def Get_image(prompt, base, refiner, n_steps, high_noise_frac, image_path):
    # Run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        height=576,
        width=1024,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        height=576,
        width=1024,
        image=image,
    ).images[0]
    image.save(image_path)
    return image


def Get_Last_Frame(video_path, output_image_path):
    # Load the video file using VideoFileClip
    with VideoFileClip(video_path) as video:
        # Get the last frame by going to the last second of the video
        last_frame = video.get_frame(video.duration - 0.01)  # a fraction before the end

    # Now, we save the last frame as an image using PIL
    last_frame_image = Image.fromarray(last_frame)
    last_frame_image.save(output_image_path)
    return last_frame_image


def generate_and_concatenate_videos(initial_image_path, pipe, num_iterations=60):
    video_paths = []  # To keep track of all generated video paths
    current_image_path = initial_image_path

    for iteration in range(num_iterations):
        # Generate frames based on the current image
        image = Image.open(current_image_path).resize((1024, 576))
        seed = int(time.time())
        torch.manual_seed(seed)
        frames = pipe(image, decode_chunk_size=12, generator=torch.Generator(), motion_bucket_id=127).frames[0]

        # Export frames to video and save the path
        video_path = f"video_segment_{iteration}.mp4"
        export_to_video(frames, video_path, fps=25)
        video_paths.append(video_path)

        # Get the last frame of the current video for the next iteration
        current_image_path = Get_Last_Frame(video_path, "1.png")

    # Load and concatenate all video segments
    clips = [VideoFileClip(path) for path in video_paths]
    final_clip = concatenate_videoclips(clips)

    # Save the final video
    final_clip.write_videofile("final_output_video.mp4")
    return "final_output_video.mp4"


if __name__ == "__main__":
    pipe, base, refiner, n_steps, high_noise_frac = load_models()

    # Example usage
    Get_image("A boy", base, refiner, n_steps, high_noise_frac)
    generate_and_concatenate_videos("image.png", pipe, 3)
