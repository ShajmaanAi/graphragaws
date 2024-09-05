import gradio as gr
import pandas as pd
import os
import subprocess
import RAG
 
# Load data from Excel file and create a dictionary for lookups
df = pd.read_excel('./RAG_total/TEST249.xlsx')
dict_data = dict(zip(df['Key'], df['Summary']))
 
def on_submit(ticket_number):
 
    summary = dict_data.get(ticket_number, "Summary not found")
    if summary is None:
        solution_from_comments = "Solution not present"

    else:
        solution_from_comments = RAG.RAG_summary(summary)

    folder="/home/devuser/Desktop/DefectAnalyser/graphrag_book_Updated"
    query = summary
    s = subprocess.run(["python", "-m","graphrag.query","--method","global",query],capture_output=True, cwd=folder, check=True, text=True)
    
    # s = subprocess.getstatusoutput(f'ps -ef | grep python3')
    #print(s)
    s=s.stdout.strip()
    issues_noticed = s
    print(issues_noticed)
    
    # solution_from_comments = ""
   
    return summary, issues_noticed, solution_from_comments

css = """#hh{margin-left: 300px} #aa{margin-top: 50px}"""
 
def create_interface():
    with gr.Blocks(css=css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(r"/home/devuser/Desktop/DefectAnalyser/graphrag_book_Updated/RAG_total/acsia.jpeg",
                         width=0, elem_id="aa", show_download_button=False,
                         show_label=False, container=False)
            with gr.Column(scale=1):
                gr.Image(r"/home/devuser/Desktop/DefectAnalyser/graphrag_book_Updated/RAG_total/bmw.png",
                         width=10, elem_id="hh", show_download_button=False,
                          show_label=False, container=False)
            with gr.Column(scale=2):
                gr.HTML("<h1 style='text-align: center;'>Defect Analyser</h1>")
        with gr.Row():
            with gr.Column():
                ticket_number = gr.Textbox(label="Enter Ticket number")
                submit_button = gr.Button("Submit")
            with gr.Column():
                with gr.Row():
                    summary = gr.Textbox(label="Summary")
                with gr.Row():
                    issues_noticed = gr.Textbox(label="Similar Issues Identified from Knowledge graph", lines=10)
                with gr.Row():
                    solution_from_comments = gr.Textbox(label="Solution pulled from comments using RAG", lines=10)
       
        # Set up button click event
        submit_button.click(fn=on_submit,inputs=[ticket_number],outputs=[summary, issues_noticed, solution_from_comments])
   
    return demo
# on_submit('HU22DM-298900')
interface = create_interface()
interface.launch(server_name="0.0.0.0")

