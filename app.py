import gradio as gr
import json
import os
from diffusers import StableDiffusionPipeline
import torch

PROMPTS_FILE = "prompts.json"

# Carregar prompts salvos
def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Salvar prompt com título e imagem
def save_prompt(title, prompt, image_path="placeholder.png"):
    prompts = load_prompts()
    prompts.append({"title": title, "prompt": prompt, "image": image_path})
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False)

# Inicializar modelo Stable Diffusion v1.5
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# Função de geração de imagem
def generate_image(prompt):
    image = pipe(prompt).images[0]
    output_path = f"renders/{prompt[:20].replace(' ', '_')}.png"
    os.makedirs("renders", exist_ok=True)
    image.save(output_path)
    return output_path

# Interface Gradio
def render_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎬 Wan Render Interface (Teste com SD v1.5)")

        prompt_input = gr.Textbox(label="Digite seu prompt")
        title_input = gr.Textbox(label="Título para salvar")
        cloud_choice = gr.Dropdown(["Local (SD v1.5)", "Wan (futuro)", "Colab", "Kaggle", "HuggingFace"], label="Escolha a nuvem")

        save_button = gr.Button("Salvar Prompt")
        output = gr.Textbox(label="Resultado")

        def handle_save(title, prompt):
            save_prompt(title, prompt)
            return f"Prompt '{title}' salvo!"

        save_button.click(handle_save, [title_input, prompt_input], output)

        # Carregar prompts salvos
        with gr.Row():
            load_button = gr.Button("Carregar Prompts")
            prompt_list = gr.Textbox(label="Prompts Salvos", interactive=False)

        load_button.click(
            fn=lambda: "\n".join([p["title"] for p in load_prompts()]),
            outputs=prompt_list
        )

        # Geração de imagem
        generate_button = gr.Button("Gerar Imagem")
        image_output = gr.Image(label="Imagem Gerada")

        generate_button.click(fn=generate_image, inputs=prompt_input, outputs=image_output)

    return demo

if __name__ == "__main__":
    render_interface().launch()
