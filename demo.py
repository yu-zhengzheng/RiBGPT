import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import assembly

def generate_backdescription(title, skill1, skill1_val, skill2, skill2_val, skill3, skill3_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/ft3"
    attr = skill1 + '+' + str(int(skill1_val)) + ', ' + skill2 + '+' + str(int(skill2_val)) + ', ' + skill3 + '+' + str(int(skill3_val))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
    input_sentence = "This is the story of [PAWN_nameDef], a " + title + " with " + attr.replace("+-","-") + ": "
    print("input:",input_sentence)
    trun = len(input_sentence)
    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids.to(device)
    length_check=0 #This is the length of the generated sequence in bytes.
    while length_check<180:
        outputs = model.generate(input_ids, do_sample=True, max_length=200, temperature=1, top_p=1)
        generate_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generate_sentence = generate_sentence[0][trun:].replace('"','').replace('\xa0'," ")
        length_check=len(generate_sentence)
        print("output length :", length_check, "\noutput: \n", generate_sentence)
    return generate_sentence


if __name__ == '__main__':
    # first assemble the model
    assembly.assemble("ft3")

    # build the interface
    with gr.Blocks() as demo:
        skills = ['artistic', 'animals', 'construction', 'cooking', 'crafting', 'intellectual', 'medicine', 'melee',
                   'mining', 'plants', 'shooting', 'social']
        title = gr.Text(label="Character Title")

        with gr.Row():
            with gr.Column(scale=3):
                skill1 = gr.Dropdown(choices=skills, label="Select a skill modifier", info="Select a skill modifier for the character")
                skill2 = gr.Dropdown(choices=skills, label="Select a skill modifier", info="Select a skill modifier for the character")
                skill3 = gr.Dropdown(choices=skills, label="Select a skill modifier", info="Select a skill modifier for the character")

            with gr.Column(scale=2):
                skill1_val = gr.Number(label="Skill modifier 1 Value", info="Please enter the value here", minimum=-9, maximum=9)
                skill2_val = gr.Number(label="Skill modifier 2 Value", info="Please enter the value here", minimum=-9, maximum=9)
                skill3_val = gr.Number(label="Skill modifier 3 Value", info="Please enter the value here", minimum=-9, maximum=9)

        button = gr.Button("Submit", variant="primary")

        output = gr.Text(label="Background Description")

        button.click(generate_backdescription, [title, skill1, skill1_val, skill2, skill2_val, skill3, skill3_val], output)


    demo.launch(inbrowser=True)
