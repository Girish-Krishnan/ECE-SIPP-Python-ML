from diffusers import StableDiffusionPipeline
import torch


def main():
    pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
    pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')

    prompt = input('Describe the image you want to create: ')
    image = pipe(prompt).images[0]
    image.save('generated.png')
    print('Saved result to generated.png')


if __name__ == '__main__':
    main()
