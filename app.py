import gradio as gr
import json
import os

PROMPTS_FILE = "prompts.json"

# Carregar prompts salvos
def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Salvar prompt com título e imagem (placeholder por enquanto)
def save_prompt(title, prompt, image_path="placeholder.png"):
    prompts = load_prompts()
    prompts.append({"title": title, "prompt": prompt, "image": image_path})
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False)

# Interface inicial
def render_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎬 Wan Render Interface")

        prompt_input = gr.Textbox(label="Digite seu prompt")
        title_input = gr.Textbox(label="Título para salvar")
        cloud_choice = gr.Dropdown(["Local", "HuggingFace", "Colab", "Kaggle"], label="Escolha a nuvem")

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

        # Placeholder para gerar imagem (vamos integrar depois)
        generate_button = gr.Button("Gerar Imagem")
        image_output = gr.Image(label="Imagem Gerada")

        def fake_generate(prompt):
            return "placeholder.png"

        generate_button.click(fn=fake_generate, inputs=prompt_input, outputs=image_output)

    return demo

if __name__ == "__main__":
    render_interface().launch()
