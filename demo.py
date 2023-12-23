import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import assembly
import os

model_name = "ft1"


def choose_model(model):
    # first assemble the model in case it's partitioned
    msg = "Selected model " + model + "\n" + assembly.assemble(model)

    global model_name
    model_name = model
    return msg


def generate_backdescription(title, skill1, skill1_val, skill2, skill2_val, skill3, skill3_val):

    # set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/" + model_name
    skill_modifier_str = skill1 + '+' + str(int(skill1_val)) + ', ' + skill2 + '+' + str(int(skill2_val)) + ', ' + skill3 + '+' + str(int(skill3_val))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)

    # set up input
    input_sentence = "This is the story of [PAWN_nameDef], a " + title + " with " + skill_modifier_str.replace("+-", "-") + ": "
    print("input:", input_sentence)
    trun = len(input_sentence)
    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids.to(device)
    length_check = 0  # This is the length of the generated sequence in bytes.

    # generate backdescriptions until it has desired length
    while length_check < 180:
        outputs = model.generate(input_ids, do_sample=True, max_length=200, temperature=1, top_p=1)
        generate_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generate_sentence = generate_sentence[0][trun:].replace('"', '').replace('\xa0', " ")
        length_check = len(generate_sentence)
        print("output length :", length_check, "\noutput: \n", generate_sentence)


    return generate_sentence


if __name__ == '__main__':
    # build the interface
    RiBGPT_theme=gr.themes.Soft(primary_hue="slate",text_size="lg",spacing_size="sm")
    with gr.Blocks(title="RiBGPT",theme=RiBGPT_theme) as demo:
        # define dropdown choices
        skills = ['artistic', 'animals', 'construction', 'cooking', 'crafting', 'intellectual', 'medicine', 'melee',
                  'mining', 'plants', 'shooting', 'social']
        models = os.listdir("models")


        # define the UI components
        gr.Image("RiBGPT.png", container=False)
        with gr.Row():
            model = gr.Dropdown(choices=models, label="Select model", info="Enter the name of he model (e.g. ft1) and press Select to select model", scale=3)
            button_assembly = gr.Button("Select", variant="primary", scale=1)
        title = gr.Text(label="Pawn Title", value="plague doctor", info="Enter any title, lower case")
        with gr.Row():
            with gr.Column(scale=3):
                skill1 = gr.Dropdown(choices=skills, label="Select a skill modifier", value="construction", info="Select a skill modifier for the pawn")
                skill2 = gr.Dropdown(choices=skills, label="Select a skill modifier", value="crafting", info="Select a skill modifier for the pawn")
                skill3 = gr.Dropdown(choices=skills, label="Select a skill modifier", value="plants", info="Select a skill modifier for the pawn")
            with gr.Column(scale=2):
                skill1_val = gr.Number(label="Skill modifier 1 Value", value=1,info="Please enter the value here", minimum=-9, maximum=9)
                skill2_val = gr.Number(label="Skill modifier 2 Value", value=2,info="Please enter the value here", minimum=-9, maximum=9)
                skill3_val = gr.Number(label="Skill modifier 3 Value", value=-1,info="Please enter the value here", minimum=-9, maximum=9)
        button_generate = gr.Button("Generate Backdescription", variant="primary")
        output = gr.Text(label="Background Description",value="\n\n\n\n")


        # define button behaviour
        button_assembly.click(choose_model, model, output)
        button_generate.click(generate_backdescription, [title, skill1, skill1_val, skill2, skill2_val, skill3, skill3_val], output)

    # launch the interface
    demo.launch(inbrowser=True)
